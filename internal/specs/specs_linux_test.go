//go:build linux

package specs

import "testing"

func TestCapCgroupMemory(t *testing.T) {
	cases := []struct{ total, avail, limit, used, wantTotal, wantAvail int }{
		{4096, 3000, 0, -1, 4096, 3000},
		{8192, 5000, 4096, 3500, 4096, 596},
		{4096, 3000, 4096, 4096, 4096, 1},
		{4096, 1200, 2048, -1, 2048, 1024},
		{4096, 600, 2048, 300, 2048, 600},
	}
	for _, c := range cases {
		total, avail := capCgroupMemory(c.total, c.avail, c.limit, c.used)
		if total != c.wantTotal || avail != c.wantAvail {
			t.Fatalf("%+v got %d/%d", c, total, avail)
		}
	}
}
