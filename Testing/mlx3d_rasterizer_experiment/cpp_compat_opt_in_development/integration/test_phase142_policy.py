import unittest
from phase142_combined import choose_sizing, choose_sort

class PolicyTests(unittest.TestCase):
    def sizing(self, prior, *, supported=True, width=1920, height=1080, n=700000):
        return choose_sizing(supported=supported,width=width,height=height,n=n,previous=prior,
                             bands=((1920,1080,700000,31_000_000,32_000_000),))

    def test_sizing_requires_history(self):
        self.assertEqual(self.sizing(None), "auto")
        self.assertEqual(self.sizing((640,360,31_000_000)), "auto")

    def test_sizing_band_edges(self):
        for count, expected in [(30_999_999,"auto"), (31_000_000,"active"),
                                (32_000_000,"active"), (32_000_001,"auto")]:
            self.assertEqual(self.sizing((1920,1080,count)), expected)

    def test_sizing_excludes_unknown_device_count_and_small(self):
        self.assertEqual(self.sizing((1920,1080,31_500_000), supported=False), "auto")
        self.assertEqual(self.sizing((1920,1080,31_500_000), n=400096), "auto")
        self.assertEqual(self.sizing((640,360,31_500_000), width=640,height=360), "auto")

    def sort(self, slots, *, width=1920,height=1080,tiles=8160,supported=True):
        return choose_sort(supported=supported,width=width,height=height,tiles=tiles,slots=slots)

    def test_sorts_according_to_actual_slots(self):
        for count, expected in [(31_999_999,"auto"), (32_000_000,"selective_final3_scatter512"),
                                (34_000_000,"selective_final3_scatter512"), (34_000_001,"auto")]:
            self.assertEqual(self.sort(count), expected)

    def test_shape_tile_and_device_fallback(self):
        self.assertEqual(self.sort(32_000_000,supported=False), "auto")
        self.assertEqual(self.sort(32_000_000,width=2560,height=1440,tiles=14400),"auto")
        self.assertEqual(self.sort(32_000_000,tiles=8193),"auto")
        self.assertEqual(self.sort(31_500_000),"auto")

if __name__ == "__main__":
    unittest.main()
