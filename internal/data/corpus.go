package data

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// TextFiles returns the .txt files under dir in a stable (sorted) order.
func TextFiles(dir string) ([]string, error) {
	var paths []string
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if !d.IsDir() && strings.HasSuffix(path, ".txt") {
			paths = append(paths, path)
		}
		return nil
	})
	sort.Strings(paths)
	return paths, err
}

// LoadTextCorpus reads every non-empty .txt file under dir. logf receives one line per file.
func LoadTextCorpus(dir string, logf func(format string, args ...any)) ([]string, error) {
	paths, err := TextFiles(dir)
	if err != nil {
		return nil, err
	}
	var docs []string
	for _, path := range paths {
		content, readErr := os.ReadFile(path)
		if readErr != nil || len(content) == 0 {
			continue
		}
		docs = append(docs, string(content))
		if logf != nil {
			logf("  Loaded corpus: %s (%d KB)\n", filepath.Base(path), len(content)/1024)
		}
	}
	if len(docs) == 0 {
		return nil, fmt.Errorf("no text documents found in %s", dir)
	}
	return docs, nil
}
