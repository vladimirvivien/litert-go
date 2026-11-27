// Command libfetch downloads the LiteRT prebuilt shared libraries for the
// target platform and prints the destination directory to stdout.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/vladimirvivien/litert-go/pkg/libfetch"
)

func main() {
	version := flag.String("version", libfetch.DefaultVersion, "LiteRT prebuilt release version")
	dir := flag.String("dir", "", "destination directory (defaults to user cache dir)")
	platform := flag.String("platform", "", "override bucket platform (e.g. linux_x86_64)")
	quiet := flag.Bool("q", false, "suppress progress logs on stderr")
	flag.Parse()

	opts := []libfetch.Option{
		libfetch.WithVersion(*version),
	}
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

	outDir, err := libfetch.Fetch(context.Background(), opts...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "libfetch: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(outDir)
}
