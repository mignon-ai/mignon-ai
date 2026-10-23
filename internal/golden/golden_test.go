package golden

// Golden behavior snapshot used to prove that refactors do not change functionality.
//
// Every case trains a tiny model single-threaded with fixed seeds (multi-threaded gradient
// accumulation is order-dependent), then fingerprints the exact float bits of weights, logits
// and serialized checkpoints. Any refactor must reproduce the snapshot bit for bit.
//
// Regenerate (only for intentional behavior changes): go test ./internal/golden -update-golden

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/mignon-ai/mignon-ai/internal/ca"
	"github.com/mignon-ai/mignon-ai/internal/checkpoint"
	"github.com/mignon-ai/mignon-ai/internal/data"
	"github.com/mignon-ai/mignon-ai/internal/infer"
	"github.com/mignon-ai/mignon-ai/internal/kernels"
	"github.com/mignon-ai/mignon-ai/internal/quant"
	"github.com/mignon-ai/mignon-ai/internal/tokenizer"
	"github.com/mignon-ai/mignon-ai/internal/train"
)

var updateGolden = flag.Bool("update-golden", false, "rewrite the golden behavior snapshot")

const (
	goldenPath       = "testdata/behavior.json"
	goldenVocab      = 24
	goldenState      = 8
	goldenHidden     = 12
	goldenSteps      = 3
	goldenDilation   = 2
	goldenSeqLen     = 16
	goldenAttnHeads  = 2
	goldenAttnWindow = 8 // shorter than goldenSeqLen, so the window cuts in
	goldenTrainSeqs  = 24
	goldenValSeqs    = 6
	goldenEpochs     = 3
	goldenBatch      = 8
	goldenModelSeed  = 7
	goldenDataSeed   = 11
	goldenOrderSeed  = 13
	goldenAppendToks = 4
	goldenTokVocab   = 300
)

// goldenModelCase is one architecture/optimizer combination to fingerprint.
type goldenModelCase struct {
	Name       string
	Blocks     int
	Tied       bool
	AttnHeads  int
	AttnWindow int
	Optimizer  string
	LR         float64
}

var goldenModelCases = []goldenModelCase{
	{Name: "muon_2block", Blocks: 2, Tied: true, Optimizer: "muon", LR: 0.04},
	{Name: "muon_2block_attn", Blocks: 2, Tied: true, AttnHeads: goldenAttnHeads, AttnWindow: goldenAttnWindow, Optimizer: "muon", LR: 0.04},
	{Name: "adam_2block_untied", Blocks: 2, Tied: false, Optimizer: "adam", LR: 0.01},
}

// goldenModelResult fingerprints everything observable about one trained model.
type goldenModelResult struct {
	InitWeights     string  `json:"init_weights"`
	BestLoss        float64 `json:"best_loss"`
	BestEpoch       int     `json:"best_epoch"`
	FinalLoss       float64 `json:"final_loss"`
	FinalTrainLoss  float64 `json:"final_train_loss"`
	TrainedWeights  string  `json:"trained_weights"`
	FloatLogits     string  `json:"float_logits"`
	QuantLogits     string  `json:"quant_logits"`
	SessionLogits   string  `json:"session_logits"`
	DequantWeights  string  `json:"dequant_weights"`
	FloatCheckpoint string  `json:"float_checkpoint"`
	QuantCheckpoint string  `json:"quant_checkpoint"`
}

type goldenTokenizerResult struct {
	Vocab     string `json:"vocab"`
	Encoded   string `json:"encoded"`
	Roundtrip bool   `json:"roundtrip"`
}

type goldenSnapshot struct {
	AVX2      bool                         `json:"avx2"`
	Tokenizer goldenTokenizerResult        `json:"tokenizer"`
	Configs   map[string]string            `json:"configs"`
	Models    map[string]goldenModelResult `json:"models"`
	Data      map[string]string            `json:"data"`
}

func TestGoldenBehavior(t *testing.T) {
	tok := goldenTokenizer()
	snap := goldenSnapshot{
		AVX2:      kernels.HasAVX2,
		Tokenizer: fingerprintTokenizer(tok),
		Configs:   fingerprintPresetConfigs(t),
		Models:    map[string]goldenModelResult{},
		Data:      fingerprintDataPipeline(t),
	}
	for _, c := range goldenModelCases {
		snap.Models[c.Name] = runGoldenModelCase(t, c, tok)
	}

	if *updateGolden {
		writeGolden(t, snap)
		return
	}
	want := readGolden(t)
	if want.AVX2 != snap.AVX2 {
		t.Skipf("golden snapshot recorded with avx2=%v, this CPU has avx2=%v", want.AVX2, snap.AVX2)
	}
	compareGolden(t, "", reflect.ValueOf(want), reflect.ValueOf(snap))
}

// TestGoldenDeterminism guards the snapshot itself: two identical runs must agree exactly.
func TestGoldenDeterminism(t *testing.T) {
	tok := goldenTokenizer()
	c := goldenModelCases[2]
	a, b := runGoldenModelCase(t, c, tok), runGoldenModelCase(t, c, tok)
	if a != b {
		t.Fatalf("golden case %s is not deterministic:\n%+v\n%+v", c.Name, a, b)
	}
}

func runGoldenModelCase(t *testing.T, c goldenModelCase, tok *tokenizer.Tokenizer) goldenModelResult {
	t.Helper()
	configs := ca.BuildBlockConfigs(c.Blocks, goldenSteps, goldenDilation)
	model := ca.New(ca.Config{VocabSize: goldenVocab, StateSize: goldenState, HiddenSize: goldenHidden, Blocks: configs, TiedEmbed: c.Tied, AttnHeads: c.AttnHeads, AttnWindow: c.AttnWindow})
	model.InitWeights(goldenModelSeed)
	var r goldenModelResult
	r.InitWeights = hashF32(model.Weights())

	trainSet := goldenSequences(goldenDataSeed, goldenTrainSeqs)
	valSet := goldenSequences(goldenDataSeed+1, goldenValSeqs)
	res := train.Run(model, trainSet, valSet, train.Options{
		Epochs:      goldenEpochs,
		LR:          c.LR,
		Optimizer:   c.Optimizer,
		BatchSize:   goldenBatch,
		WeightDecay: 0.0001,
		Threads:     1,
		Seed:        goldenOrderSeed,
		Quiet:       true,
	})
	r.BestLoss, r.BestEpoch, r.FinalLoss, r.FinalTrainLoss = res.BestLoss, res.BestEpoch, res.FinalLoss, res.FinalTrainLoss
	r.TrainedWeights = hashF32(model.Weights())

	window := valSet.X[0]
	r.FloatLogits = hashF64(model.Forward(window))

	qm := quant.Quantize(model, goldenSeqLen)
	r.QuantLogits = hashF64(qm.Forward(window))
	r.SessionLogits = fingerprintSession(qm.NewSession(goldenSeqLen), window)
	r.DequantWeights = hashF32(quant.Dequantize(qm).Weights())

	meta := checkpoint.Metadata{
		Blocks: c.Blocks, StateSize: goldenState, HiddenSize: goldenHidden,
		Steps: goldenSteps, DilationBase: goldenDilation, ContextSize: goldenSeqLen, Task: "text",
		TiedEmbed: c.Tied, AttnHeads: c.AttnHeads, AttnWindow: c.AttnWindow,
	}
	dir := t.TempDir()
	r.FloatCheckpoint = hashFile(t, dir, "float.json", func(p string) error { return checkpoint.Save(model, tok, meta, p) })
	r.QuantCheckpoint = hashFile(t, dir, "quant.json", func(p string) error { return quant.Save(qm, tok, meta, p) })
	return r
}

// goldenSequences builds a learnable synthetic corpus: each sequence follows a noisy affine
// recurrence over the vocabulary, with masked padding at the end of some sequences.
func goldenSequences(seed int64, n int) (set data.Sequences) {
	rng := rand.New(rand.NewSource(seed))
	for i := 0; i < n; i++ {
		step := 1 + rng.Intn(3)
		tokens := make([]int, goldenSeqLen+1)
		tokens[0] = rng.Intn(goldenVocab)
		for j := 1; j < len(tokens); j++ {
			tokens[j] = (tokens[j-1] + step) % goldenVocab
			if rng.Intn(8) == 0 {
				tokens[j] = rng.Intn(goldenVocab)
			}
		}
		x := append([]int(nil), tokens[:goldenSeqLen]...)
		y := append([]int(nil), tokens[1:]...)
		if i%4 == 3 {
			for j := goldenSeqLen - 3; j < goldenSeqLen; j++ {
				y[j] = data.MaskedTarget
			}
		}
		set.X, set.Y = append(set.X, x), append(set.Y, y)
	}
	return set
}

func fingerprintSession(sess infer.Session, window []int) string {
	h := sha256.New()
	prefix := len(window) - goldenAppendToks
	writeF64(h, sess.Prefill(window[:prefix]))
	for _, tok := range window[prefix:] {
		writeF64(h, sess.Append(tok))
	}
	return hex.EncodeToString(h.Sum(nil))
}

const goldenTokenizerText = "The cellular automaton updates every cell from its neighbours. " +
	"Each rule is learned, not written by hand; tokens flow left to right.\n" +
	"git commit -m \"fix: handle rare characters like é, ß and 日本\" && git push origin main\n" +
	"func main() { fmt.Println(\"hello, world\") } // 12345 + 678 = 13023\n"

func goldenTokenizer() *tokenizer.Tokenizer {
	tok := tokenizer.New()
	tok.Train(strings.Repeat(goldenTokenizerText, 3), goldenTokVocab)
	return tok
}

func fingerprintTokenizer(tok *tokenizer.Tokenizer) goldenTokenizerResult {
	vocab, _ := json.Marshal(struct {
		V []string
		M []tokenizer.Merge
	}{tok.Vocab, tok.Merges})
	probe := goldenTokenizerText + "unseen wörds ✓ and <|eos|> markers"
	ids := tok.Encode(probe)
	enc, _ := json.Marshal(ids)
	return goldenTokenizerResult{
		Vocab:     hashBytes(vocab),
		Encoded:   hashBytes(enc),
		Roundtrip: tok.Decode(tok.Encode(goldenTokenizerText)) == goldenTokenizerText,
	}
}

// fingerprintPresetConfigs records how every preset resolves into a config and training options.
func fingerprintPresetConfigs(t *testing.T) map[string]string {
	t.Helper()
	out := map[string]string{}
	names := make([]string, 0, len(train.Presets))
	for name := range train.Presets {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range append(names, "") {
		layer := json.RawMessage(fmt.Sprintf(`{"preset":%q}`, name))
		if name == "" {
			layer = nil
		}
		cfg, err := train.ResolveConfig(layer)
		if err != nil {
			t.Fatalf("resolve preset %q: %v", name, err)
		}
		model := cfg.BuildModel(cfg.Vocab, goldenModelSeed)
		out["preset:"+name] = hashBytes([]byte(fmt.Sprintf("%+v|%+v|%d|%s",
			cfg, cfg.Options(goldenOrderSeed, true), model.NumParams(), hashF32(model.Weights()))))
	}
	return out
}

func hashF32(v []float32) string {
	h := sha256.New()
	var b [4]byte
	for _, x := range v {
		bits := math.Float32bits(x)
		b[0], b[1], b[2], b[3] = byte(bits), byte(bits>>8), byte(bits>>16), byte(bits>>24)
		h.Write(b[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}

func writeF64(h interface{ Write([]byte) (int, error) }, v []float64) {
	var b [8]byte
	for _, x := range v {
		bits := math.Float64bits(x)
		for i := range b {
			b[i] = byte(bits >> (8 * i))
		}
		h.Write(b[:])
	}
}

func hashF64(v []float64) string {
	h := sha256.New()
	writeF64(h, v)
	return hex.EncodeToString(h.Sum(nil))
}

func hashBytes(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func hashFile(t *testing.T, dir, name string, save func(path string) error) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := save(path); err != nil {
		t.Fatalf("save %s: %v", name, err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", name, err)
	}
	return hashBytes(data)
}

func writeGolden(t *testing.T, snap goldenSnapshot) {
	t.Helper()
	data, err := json.MarshalIndent(snap, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(goldenPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(goldenPath, append(data, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("wrote %s", goldenPath)
}

func readGolden(t *testing.T) goldenSnapshot {
	t.Helper()
	data, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatalf("missing golden snapshot (run with -update-golden on the pre-refactor code): %v", err)
	}
	var snap goldenSnapshot
	if err := json.Unmarshal(data, &snap); err != nil {
		t.Fatal(err)
	}
	return snap
}

// compareGolden reports every differing leaf so a failure points at the exact behavior that changed.
func compareGolden(t *testing.T, path string, want, got reflect.Value) {
	t.Helper()
	switch want.Kind() {
	case reflect.Struct:
		for i := 0; i < want.NumField(); i++ {
			compareGolden(t, path+"."+want.Type().Field(i).Name, want.Field(i), got.Field(i))
		}
	case reflect.Map:
		keys := map[string]bool{}
		for _, k := range append(want.MapKeys(), got.MapKeys()...) {
			keys[k.String()] = true
		}
		for k := range keys {
			kv := reflect.ValueOf(k)
			w, g := want.MapIndex(kv), got.MapIndex(kv)
			if !w.IsValid() || !g.IsValid() {
				t.Errorf("%s[%s]: present in golden=%v, present now=%v", path, k, w.IsValid(), g.IsValid())
				continue
			}
			compareGolden(t, path+"["+k+"]", w, g)
		}
	default:
		if !reflect.DeepEqual(want.Interface(), got.Interface()) {
			t.Errorf("%s: golden=%v, now=%v", path, want.Interface(), got.Interface())
		}
	}
}
