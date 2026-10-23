package eval

import "testing"

func TestClassifyFailure(t *testing.T) {
	branch := Task{
		Instruction: "pull rebase from origin experiment/email-notifications",
		Expected:    []string{"git pull --rebase origin experiment/email-notifications"},
		Complete:    true,
	}
	// A second pull request makes git, pull, --rebase and origin answer vocabulary.
	otherPull := Task{
		Instruction: "pull rebase from origin feature/login",
		Expected:    []string{"git pull --rebase origin feature/login"},
		Complete:    true,
	}
	option := Task{Instruction: "list all branches", Prompt: "git branch", Expected: []string{" -a"}}
	message := Task{Instruction: "commit 'add payment tests'", Prompt: "git commit -m '", Expected: []string{"add"}}
	classifier := newFailureClassifier([]Task{branch, otherPull, option, message})

	cases := []struct {
		name      string
		task      Task
		predicted string
		want      FailureKind
	}{
		{"wrong request word", branch, "git pull --rebase origin origin", FailWrongSpan},
		{"broken name", branch, "git pull --rebase origin experiment/email-notif", FailPartialCopy},
		{"invented name", branch, "git pull --rebase origin main", FailNoCopy},
		{"name from another request", branch, "git pull --rebase origin feature/login", FailNoCopy},
		{"wrong subcommand", branch, "git fetch --rebase origin experiment/email-notifications", FailKnowledge},
		{"wrong option", option, " -r", FailKnowledge},
		{"did not stop", branch, "git pull --rebase origin experiment/email-notifications now", FailEnding},
		{"quoted copy", message, "fix", FailNoCopy},
		{"one shared letter is not a copy", message, "a", FailNoCopy},
		{"glued onto the prompt", option, "-r", FailKnowledge},
		{"stopped before copying", message, "", FailNoCopy},
	}
	for _, c := range cases {
		got := classifier.classify(TaskResult{Task: c.task, Predicted: c.predicted})
		if got != c.want {
			t.Errorf("%s: classify = %q, want %q", c.name, got, c.want)
		}
	}
}

func TestClassifyFailureUsesClosestExpectedAnswer(t *testing.T) {
	task := Task{
		Instruction: "stage file a.go",
		Expected:    []string{"git add a.go", "git stage a.go"},
	}
	// Follows the second answer until the file name, so the copy is what failed.
	got := newFailureClassifier([]Task{task}).classify(TaskResult{Task: task, Predicted: "git stage b.go"})
	if got != FailNoCopy {
		t.Errorf("classify = %q, want %q", got, FailNoCopy)
	}
}
