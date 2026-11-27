.PHONY: fix fmt build vet test test-integration bench lint tidy clean all

# Default target: CI-equivalent local verification.
all: fix fmt build vet test lint

fix:
	go fix ./...

fmt:
	go fmt ./...

build:
	go build ./...

vet:
	go vet ./...

test:
	go test -race ./...

test-integration:
	go test -race -tags=integration ./...

bench:
	go test -bench=. -benchmem ./...

lint:
	golangci-lint run ./...

tidy:
	go mod tidy

clean:
	go clean ./...
