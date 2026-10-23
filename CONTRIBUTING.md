# Contributing

Issues and pull requests are welcome.

- `go vet ./... && go test ./...` must pass; `gofmt -l .` must print nothing. The tests run in a few seconds.
- Changes to `packages/mignon-web` need `npm test` there (needs Go and Node 18+).
- Keep changes focused, and explain results you claim for model quality with the command that reproduces them.

By submitting a contribution you agree that it is licensed under the project's [Elastic License 2.0](LICENSE) and that
the maintainers may also relicense it as part of the project's future releases.
