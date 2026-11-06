.PHONY: windows linux test clean

windows:
	CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build -trimpath -ldflags="-s -w" -o matrix.exe ./cmd/matrix

linux:
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags="-s -w" -o matrix ./cmd/matrix

test:
	go test -v ./...

clean:
	rm -f matrix-linux
