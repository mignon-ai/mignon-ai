package tokenizer

import (
	"os"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestTokenizerTrainAndRoundTrip(t *testing.T) {
	sampleText := `hello world! hello there. how are you doing today? 
def add(a, b):
    return a + b
speak speak speak to me, citizen!`

	tok := New()
	targetVocab := 60
	tok.Train(sampleText, targetVocab)

	if tok.VocabSize() == 0 {
		t.Fatalf("Expected non-zero vocabulary size")
	}

	testPhrases := []string{
		"hello world!",
		"def add(a, b):",
		"return a + b",
		"speak, citizen!",
	}

	for _, phrase := range testPhrases {
		encoded := tok.Encode(phrase)
		decoded := tok.Decode(encoded)

		if decoded != phrase {
			t.Errorf("Round-trip failed for '%s'. Got '%s'", phrase, decoded)
		}

		// Verify subword compression: token count should be <= character count
		if len(encoded) > len(phrase) {
			t.Errorf("Expected token count <= character count, got %d > %d", len(encoded), len(phrase))
		}
	}
}

func TestTokenizerSaveAndLoad(t *testing.T) {
	sampleText := "to be or not to be, that is the question: whether 'tis nobler in the mind"
	tok := New()
	tok.Train(sampleText, 50)

	tmpFile := "scratch_test_tok.json"
	defer os.Remove(tmpFile)

	err := tok.Save(tmpFile)
	if err != nil {
		t.Fatalf("Failed to save tokenizer: %v", err)
	}

	loadedTok, err := Load(tmpFile)
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	if loadedTok.VocabSize() != tok.VocabSize() {
		t.Errorf("Vocab size mismatch: %d vs %d", loadedTok.VocabSize(), tok.VocabSize())
	}

	phrase := "that is the question"
	originalEncoded := tok.Encode(phrase)
	loadedEncoded := loadedTok.Encode(phrase)

	if len(originalEncoded) != len(loadedEncoded) {
		t.Fatalf("Encoded length mismatch: %d vs %d", len(originalEncoded), len(loadedEncoded))
	}

	for i := range originalEncoded {
		if originalEncoded[i] != loadedEncoded[i] {
			t.Errorf("Token mismatch at %d: %d vs %d", i, originalEncoded[i], loadedEncoded[i])
		}
	}

	decoded := loadedTok.Decode(loadedEncoded)
	if decoded != phrase {
		t.Errorf("Decoded mismatch: expected '%s', got '%s'", phrase, decoded)
	}
}

func TestShakespeareBPE(t *testing.T) {
	data, err := os.ReadFile("../../data/shakespeare/romeo_snippet.txt")
	if err != nil {
		t.Skip("Shakespeare file not found, skipping")
	}

	text := string(data)
	tok := New()
	tok.Train(text, 256)

	encoded := tok.Encode(text)
	compressionRatio := float64(len(text)) / float64(len(encoded))

	t.Logf("Raw characters: %d, BPE tokens: %d, Compression: %.2fx", len(text), len(encoded), compressionRatio)
	t.Logf("Vocabulary size: %d", tok.VocabSize())

	// Top learned merges:
	topCount := 10
	if len(tok.Merges) < topCount {
		topCount = len(tok.Merges)
	}
	for i := 0; i < topCount; i++ {
		t.Logf("Merge %2d: '%s' + '%s' -> '%s'", i+1, tok.Merges[i].Pair1, tok.Merges[i].Pair2, tok.Merges[i].Pair1+tok.Merges[i].Pair2)
	}

	// Verify exact lossless roundtrip on entire 8.8KB file
	decoded := tok.Decode(encoded)
	if decoded != text {
		t.Errorf("Lossless reconstruction failed on Shakespeare text")
	}
}

func TestTokenizerUnkDecoding(t *testing.T) {
	tok := New()
	tok.Train("hello world", 20)

	// 'z' was not in training text, should decode to replacement character
	encoded := tok.Encode("z")
	decoded := tok.Decode(encoded)
	if decoded != "\uFFFD" {
		t.Fatalf("Expected replacement character for UNK token, got %q", decoded)
	}
}

func TestTokenizerSkipsRareCharacters(t *testing.T) {
	// Large enough that a single occurrence falls under MinCharsPerMillion.
	text := strings.Repeat("git commit pushes the branch. ", 80000) + "rocket 🚀 done"
	if minCharFreq(utf8.RuneCountInString(text)) <= 1 {
		t.Fatalf("test text too small to exercise the rare-character threshold")
	}
	tok := New()
	tok.Train(text, 64)

	if _, ok := tok.TokenToID["🚀"]; ok {
		t.Errorf("rare character got a vocab slot")
	}
	for _, v := range tok.Vocab {
		if strings.ContainsRune(v, '🚀') {
			t.Errorf("rare character appears in merged token %q", v)
		}
	}
	ids := tok.Encode("🚀")
	if len(ids) != 1 || ids[0] != tok.TokenToID[UnkToken] {
		t.Errorf("rare character should encode as <UNK>, got %v", ids)
	}
	if got := tok.Decode(tok.Encode("git commit")); got != "git commit" {
		t.Errorf("frequent text should round-trip, got %q", got)
	}
}

func TestMinCharFreqKeepsSmallSamples(t *testing.T) {
	if minCharFreq(0) != 1 || minCharFreq(int(charsPerMillion)-1) != 1 {
		t.Errorf("samples under a million characters must keep every character")
	}
}

func TestSpecialTokensAtomicEncoding(t *testing.T) {
	tok := New()
	tok.Train("delete branch fix-auth commit hash 1234 main.go", 50)

	tok.AddSpecialToken("<branch>")
	tok.AddSpecialToken("</branch>")
	tok.AddSpecialToken("<file>")

	// Ensure slot tokens exist in vocabulary
	branchOpenID, ok1 := tok.TokenToID["<branch>"]
	branchCloseID, ok2 := tok.TokenToID["</branch>"]
	fileID, ok3 := tok.TokenToID["<file>"]
	if !ok1 || !ok2 || !ok3 {
		t.Fatalf("Expected slot tokens to be registered in tokenizer vocabulary")
	}

	text := "git branch -d <branch>fix-auth</branch> <file>main.go</file>"
	tokens := tok.Encode(text)

	// Verify that <branch> and </branch> are encoded as single atomic token IDs
	hasBranchOpen := false
	hasBranchClose := false
	hasFile := false
	for _, id := range tokens {
		if id == branchOpenID {
			hasBranchOpen = true
		}
		if id == branchCloseID {
			hasBranchClose = true
		}
		if id == fileID {
			hasFile = true
		}
	}

	if !hasBranchOpen || !hasBranchClose {
		t.Errorf("Expected atomic <branch> and </branch> tokens in encoded sequence, got %v", tokens)
	}
	if !hasFile {
		t.Errorf("Expected atomic <file> token in encoded sequence, got %v", tokens)
	}
}

func TestPadTokenID(t *testing.T) {
	tok := New()
	if got, want := tok.PadTokenID(), tok.TokenToID[PadToken]; got != want {
		t.Fatalf("PadTokenID() = %d, want %d", got, want)
	}
	var missing *Tokenizer
	if missing.PadTokenID() != 0 {
		t.Fatalf("nil tokenizer should pad with 0")
	}
}
