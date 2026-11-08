/* The codec catalog, in the order of the cards in docs.html. No page loads this
   file: the prev and next links on each codec page are static HTML. test_catalog.py
   reads the list and checks it against docs.html, the README table and the codec
   pages. */
const CODECS = [
  { no: "01", name: "delta_w",      file: "delta-w.html",      fam: "purple" },
  { no: "02", name: "delta_n",      file: "delta-n.html",      fam: "purple" },
  { no: "03", name: "planar",       file: "planar.html",       fam: "purple" },
  { no: "04", name: "med",          file: "med.html",          fam: "purple" },
  { no: "05", name: "average",      file: "average.html",      fam: "purple" },
  { no: "06", name: "wp_static",    file: "wp-static.html",    fam: "purple" },
  { no: "07", name: "planar_zigzag", file: "planar-zigzag.html", fam: "gold" },
  { no: "08", name: "planar_zigzag_pfor", file: "planar-zigzag-pfor.html", fam: "gold" },
  { no: "09", name: "planar_zigzag_lanes", file: "planar-zigzag-lanes.html", fam: "gold" },
  { no: "10", name: "med_zigzag",   file: "med-zigzag.html",   fam: "gold" },
  { no: "11", name: "deinterleave", file: "deinterleave.html", fam: "green"  },
  { no: "12", name: "nodata",       file: "nodata.html",       fam: "green"  },
  { no: "13", name: "pfor",         file: "pfor.html",         fam: "green"  },
  { no: "14", name: "lanes", file: "lanes.html", fam: "green" },
  { no: "15", name: "quant_linear", file: "quant-linear.html", fam: "orange" },
  { no: "16", name: "quant_log",    file: "quant-log.html",    fam: "orange" },
  { no: "17", name: "quant_sqrt",   file: "quant-sqrt.html",   fam: "orange" }
];
