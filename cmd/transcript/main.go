// Command transcript serves the local UI for the Turkish transcript pipeline.
//
// The UI is a web page on 127.0.0.1 rather than a native window: no GUI
// toolkit means no cgo, which means this cross-compiles to a single Windows
// executable from any machine.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"syscall"
	"time"

	"aitranscript/internal/audio"
	"aitranscript/internal/web"
)

func main() {
	var (
		addr        = flag.String("addr", "127.0.0.1:8765", "address to listen on; the port falls back to a free one if taken")
		audioDir    = flag.String("audio", "audios", "default audio library folder")
		videoDir    = flag.String("video", "videos", "default video library folder")
		outputDir   = flag.String("output", filepath.Join("work", "wav"), "directory to write WAV files to")
		modelDir    = flag.String("models", "models", "directory holding ggml models")
		transcripts = flag.String("transcripts", filepath.Join("work", "transcripts"), "directory to write transcripts to")
		datasetDir  = flag.String("dataset", filepath.Join("work", "dataset"), "directory for reviewed corrections")
		stateDir    = flag.String("state", "work", "directory for the job queue, settings and glossary")
		concurrency = flag.Int("jobs", 2, "how many ffmpeg processes may run at once")
		noOpen      = flag.Bool("no-open", false, "do not open a browser on startup")
	)
	flag.Parse()

	tools, err := audio.Locate()
	if err != nil {
		log.Fatalf("ffmpeg is required but was not found.\n  %v\n\n"+
			"On Linux install it from your package manager (e.g. 'pacman -S ffmpeg').\n"+
			"On Windows the bundled copy lives in binaries/windows/.\n"+
			"Or point at it directly with %s=/path/to/ffmpeg.", err, audio.EnvFFmpeg)
	}

	// The two library folders are the program's defaults, so create them rather
	// than refusing to start; a first run should just work.
	dirs := map[string]*string{
		"audio": audioDir, "video": videoDir, "output": outputDir,
		"models": modelDir, "transcripts": transcripts, "dataset": datasetDir,
		"state": stateDir,
	}
	abs := map[string]string{}
	for label, p := range dirs {
		a, err := filepath.Abs(*p)
		if err != nil {
			log.Fatalf("%s folder: %v", label, err)
		}
		if err := os.MkdirAll(a, 0o755); err != nil {
			log.Fatalf("creating %s folder %s: %v", label, a, err)
		}
		abs[label] = a
	}

	ln, err := listen(*addr)
	if err != nil {
		log.Fatalf("cannot listen on %s: %v", *addr, err)
	}

	srv := &http.Server{
		Handler: web.NewServer(tools, web.Config{
			AudioDir:      abs["audio"],
			VideoDir:      abs["video"],
			OutputDir:     abs["output"],
			ModelDir:      abs["models"],
			TranscriptDir: abs["transcripts"],
			DatasetDir:    abs["dataset"],
			StateDir:      abs["state"],
			Concurrency:   *concurrency,
		}).Handler(),
		// No write timeout: the SSE stream is deliberately long-lived.
		ReadHeaderTimeout: 10 * time.Second,
	}

	url := "http://" + ln.Addr().String()
	fmt.Printf("ffmpeg   %s\n", tools.FFmpeg)
	if tools.FFprobe == "" {
		fmt.Printf("ffprobe  not found (conversions run without a progress percentage)\n")
	} else {
		fmt.Printf("ffprobe  %s\n", tools.FFprobe)
	}
	fmt.Printf("audio    %s\nvideo    %s\noutput   %s\nmodels   %s\n\n",
		abs["audio"], abs["video"], abs["output"], abs["models"])
	fmt.Printf("UI ready at %s\nPress Ctrl+C to stop.\n", url)

	if !*noOpen {
		if err := openBrowser(url); err != nil {
			fmt.Printf("(could not open a browser automatically: %v)\n", err)
		}
	}

	errc := make(chan error, 1)
	go func() { errc <- srv.Serve(ln) }()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	select {
	case err := <-errc:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("server: %v", err)
		}
	case <-ctx.Done():
		fmt.Println("\nshutting down")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdownCtx)
	}
}

// listen binds addr, falling back to an arbitrary free port if it is in use so
// that a second run, or an unrelated service on 8765, is not a hard failure.
func listen(addr string) (net.Listener, error) {
	ln, err := net.Listen("tcp", addr)
	if err == nil {
		return ln, nil
	}
	host, _, splitErr := net.SplitHostPort(addr)
	if splitErr != nil {
		return nil, err
	}
	fmt.Printf("%s is in use, picking a free port instead\n", addr)
	return net.Listen("tcp", net.JoinHostPort(host, "0"))
}

func openBrowser(url string) error {
	switch runtime.GOOS {
	case "windows":
		return exec.Command("rundll32", "url.dll,FileProtocolHandler", url).Start()
	case "darwin":
		return exec.Command("open", url).Start()
	default:
		return exec.Command("xdg-open", url).Start()
	}
}
