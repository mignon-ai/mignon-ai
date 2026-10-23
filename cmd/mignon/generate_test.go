package main

import (
	"testing"

	"github.com/mignon-ai/mignon-ai/internal/generate"
)

// parseGenTestFlags parses 'mignon generate' flags.
func parseGenTestFlags(t *testing.T, args ...string) *genOptions {
	t.Helper()
	o, err := parseGenFlags(newTestFlagSet(), append([]string{"-prompt", "hi"}, args...), false)
	if err != nil {
		t.Fatalf("parseGenFlags(%v): %v", args, err)
	}
	return o
}

func TestGenerationModeFollowsTaskUnlessForced(t *testing.T) {
	cases := []struct {
		flag, task, want string
	}{
		{modeAuto, taskChat, modeChat},
		{modeAuto, taskText, modeText},
		{modeText, taskChat, modeText},
		{modeChat, taskText, modeChat},
	}
	for _, c := range cases {
		o := parseGenTestFlags(t, "-mode", c.flag)
		if got := generationMode(o, &session{task: c.task}); got != c.want {
			t.Errorf("-mode %s on a %s model: got %s, want %s", c.flag, c.task, got, c.want)
		}
	}
}

func TestUnknownModeIsRejected(t *testing.T) {
	if _, err := parseGenFlags(newTestFlagSet(), []string{"-prompt", "hi", "-mode", "story"}, false); err == nil {
		t.Fatal("expected an error for an unknown -mode")
	}
}

func TestChatSamplingDefaultsYieldToExplicitFlags(t *testing.T) {
	o := parseGenTestFlags(t)
	chat := sampling(o, modeChat)
	if chat.MaxTokens != generate.ChatMaxResponseTokens || chat.Temperature != chatTemp || chat.RepPenalty != chatPenalty {
		t.Fatalf("chat defaults: got %+v", chat)
	}
	if text := sampling(o, modeText); text.MaxTokens != defaultGenTokens || text.Temperature != defaultTemp || text.RepPenalty != defaultPenalty {
		t.Fatalf("text defaults: got %+v", text)
	}

	o = parseGenTestFlags(t, "-gen-tokens", "200", "-temp", "0", "-penalty", "1")
	if got := sampling(o, modeChat); got.MaxTokens != 200 || got.Temperature != 0 || got.RepPenalty != 1 {
		t.Fatalf("explicit flags must override chat defaults, got %+v", got)
	}
}
