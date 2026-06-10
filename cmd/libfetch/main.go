// Command libfetch downloads the LiteRT runtime libraries (libLiteRt, the
// platform accelerator, and on Windows the DirectX Shader Compiler) and
// prints the directory holding them — usable as LITERT_LIB:
//
//	LITERT_LIB=$(go run github.com/vladimirvivien/litert-go/cmd/libfetch)
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"

	"github.com/vladimirvivien/litert-go/libfetch"
)

func main() {
	version := flag.String("version", libfetch.DefaultVersion, "LiteRT prebuilt release (e.g. 2.1.5, latest)")
	dir := flag.String("dir", "", "destination directory (default: user cache dir)")
	platform := flag.String("platform", "", "bucket platform name (default: current machine)")
	quiet := flag.Bool("q", false, "suppress progress output")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	opts := []libfetch.Option{libfetch.WithVersion(*version)}
	if *dir != "" {
		opts = append(opts, libfetch.WithDir(*dir))
	}
	if *platform != "" {
		opts = append(opts, libfetch.WithPlatform(*platform))
	}
	if !*quiet {
		opts = append(opts, libfetch.WithLogf(func(format string, args ...any) {
			fmt.Fprintf(os.Stderr, format+"\n", args...)
		}))
	}

	out, err := libfetch.Fetch(ctx, opts...)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(out)
}
