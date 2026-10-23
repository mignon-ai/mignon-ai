//go:build js && wasm

// Command mignon-web is the WebAssembly build of the engine's inference side. It is the engine of the
// @mignon-ai/mignon-web package (packages/mignon-web), whose JavaScript wrapper is what pages call. It
// registers four functions on the global object:
//
//	mignonLoad(bytes: Uint8Array)                    -> JSON {"id", "info"} or {"error"}
//	mignonGenerate(id, prompt: string, maxTokens)    -> JSON {"text", "tokens": [{"text", "prob"}]} or {"error"}
//	mignonFree(id)                                   -> forgets a model
//	mignonVersion                                    -> the engine version string
//
// Several models can be loaded at once; each load returns an id. A reply is the model's greedy answer
// (see pkg/infer). Everything runs in the page; nothing is sent anywhere. What a reply means is up to the
// code around it: see examples/order-search.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"syscall/js"

	"github.com/mignon-ai/mignon-ai/pkg/infer"
	"github.com/mignon-ai/mignon-ai/pkg/model"
)

// version is set at build time: -ldflags "-X main.version=1.2.3".
var version = "dev"

var (
	models = map[int]*model.Model{}
	nextID = 1
)

// reply encodes v as the JSON string handed back to JavaScript.
func reply(v any) string {
	raw, err := json.Marshal(v)
	if err != nil {
		return `{"error":` + fmt.Sprintf("%q", err.Error()) + `}`
	}
	return string(raw)
}

func failure(err any) string { return reply(map[string]string{"error": fmt.Sprint(err)}) }

func load(_ js.Value, args []js.Value) (out any) {
	defer func() {
		if r := recover(); r != nil {
			out = failure(r)
		}
	}()
	raw := make([]byte, args[0].Get("length").Int())
	js.CopyBytesToGo(raw, args[0])
	m, err := model.Decode(raw)
	if err != nil {
		return failure(err)
	}
	id := nextID
	nextID++
	models[id] = m
	return reply(map[string]any{"id": id, "info": m.Info(), "extra": m.Extra()})
}

func generateReply(_ js.Value, args []js.Value) (out any) {
	defer func() {
		if r := recover(); r != nil {
			out = failure(r)
		}
	}()
	m, ok := models[args[0].Int()]
	if !ok {
		return failure("unknown model: it was never loaded or has been freed")
	}
	res, err := infer.Generate(context.Background(), m, args[1].String(), infer.Options{MaxTokens: args[2].Int()})
	if err != nil {
		return failure(err)
	}
	return reply(res)
}

func free(_ js.Value, args []js.Value) any {
	delete(models, args[0].Int())
	return nil
}

func main() {
	g := js.Global()
	g.Set("mignonLoad", js.FuncOf(load))
	g.Set("mignonGenerate", js.FuncOf(generateReply))
	g.Set("mignonFree", js.FuncOf(free))
	g.Set("mignonVersion", version)
	select {} // keep the runtime alive for the callbacks
}
