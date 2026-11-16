package pretrained

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

func fileSHA(path string) (string, error) {
	f, e := os.Open(path)
	if e != nil {
		return "", e
	}
	defer f.Close()
	h := sha256.New()
	if _, e = io.Copy(h, f); e != nil {
		return "", e
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
func checkSHA(path, expected string) error {
	got, err := fileSHA(path)
	if err != nil {
		return err
	}
	if got != expected {
		return fmt.Errorf("SHA-256 mismatch")
	}
	return nil
}

func downloadVerified(ctx context.Context, url, dest, sha string, max int64) error {
	if checkSHA(dest, sha) == nil {
		return nil
	}
	tmp, e := os.CreateTemp(filepath.Dir(dest), "download-*.tmp")
	if e != nil {
		return e
	}
	defer os.Remove(tmp.Name())
	defer tmp.Close()
	client := http.Client{Transport: &http.Transport{Proxy: http.ProxyFromEnvironment, ResponseHeaderTimeout: 60 * time.Second}, CheckRedirect: func(req *http.Request, via []*http.Request) error {
		if len(via) > 10 || req.URL.Scheme != "https" {
			return fmt.Errorf("unsafe redirect")
		}
		return nil
	}}
	req, e := http.NewRequestWithContext(ctx, "GET", url, nil)
	if e != nil {
		return e
	}
	resp, e := client.Do(req)
	if e != nil {
		return e
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download returned %s", resp.Status)
	}
	h := sha256.New()
	n, e := io.Copy(io.MultiWriter(tmp, h), io.LimitReader(resp.Body, max+1))
	if e != nil {
		return e
	}
	if n > max {
		return fmt.Errorf("download exceeds size limit")
	}
	if hex.EncodeToString(h.Sum(nil)) != sha {
		return fmt.Errorf("download failed SHA-256 verification")
	}
	if e = tmp.Close(); e != nil {
		return e
	}
	return os.Rename(tmp.Name(), dest)
}

// extractRuntime extracts only known executables and shared libraries, rejecting
// traversal, duplicate entries, unexpected links and excessive expanded sizes.
func extractRuntime(archive, dest string) error {
	if _, e := os.Stat(dest); e == nil {
		return fmt.Errorf("runtime directory already exists: %s; move it aside and retry", dest)
	} else if !os.IsNotExist(e) {
		return e
	}
	f, e := os.Open(archive)
	if e != nil {
		return e
	}
	defer f.Close()
	g, e := gzip.NewReader(f)
	if e != nil {
		return e
	}
	defer g.Close()
	r := tar.NewReader(g)
	stage, e := os.MkdirTemp(filepath.Dir(dest), "runtime-install-")
	if e != nil {
		return e
	}
	defer os.RemoveAll(stage)
	seen := map[string]bool{}
	links := map[string]string{}
	var total int64
	for {
		h, e := r.Next()
		if e == io.EOF {
			break
		}
		if e != nil {
			return e
		}
		name := filepath.Base(h.Name)
		wanted := name == "llama-cli" || name == "llama-finetune-lora" || strings.HasPrefix(name, "libllama.so") || strings.HasPrefix(name, "libggml") && strings.Contains(name, ".so")
		if !wanted {
			continue
		}
		clean := filepath.Clean(h.Name)
		if filepath.IsAbs(h.Name) || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
			return fmt.Errorf("unsafe runtime archive path %q", h.Name)
		}
		if seen[name] {
			return fmt.Errorf("duplicate runtime file %s", name)
		}
		seen[name] = true
		path := filepath.Join(stage, name)
		switch h.Typeflag {
		case tar.TypeReg, tar.TypeRegA:
			if h.Size < 0 || h.Size > 80<<20 {
				return fmt.Errorf("runtime file too large")
			}
			total += h.Size
			if total > 180<<20 {
				return fmt.Errorf("runtime archive too large")
			}
			mode := os.FileMode(0644)
			if name == "llama-cli" || name == "llama-finetune-lora" {
				mode = 0755
			}
			out, e := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, mode)
			if e != nil {
				return e
			}
			_, e = io.Copy(out, r)
			closeErr := out.Close()
			if e != nil {
				return e
			}
			if closeErr != nil {
				return closeErr
			}
		case tar.TypeSymlink, tar.TypeLink:
			if filepath.Base(h.Linkname) != h.Linkname || strings.HasPrefix(h.Linkname, ".") {
				return fmt.Errorf("unsafe runtime link")
			}
			links[name] = h.Linkname
		default:
			return fmt.Errorf("unsupported runtime entry %s", name)
		}
	}
	for name, target := range links {
		if !seen[target] {
			return fmt.Errorf("missing runtime link target %s", target)
		}
		if e = os.Symlink(target, filepath.Join(stage, name)); e != nil {
			return e
		}
	}
	for _, name := range []string{"llama-cli", "llama-finetune-lora"} {
		if !seen[name] {
			return fmt.Errorf("runtime archive missing %s", name)
		}
	}
	return os.Rename(stage, dest)
}
