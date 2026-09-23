# Input schema

One UTF-8 CSV per well, with the header below. Do not place original data in a public repository.

```text
well_id,depth,segment_id,interval_id,class_id,MSFL,LLS,LLD,DEN,DT,GR,NPHI
```

| Field | Meaning and constraints |
| --- | --- |
| `well_id` | One identifier per file; unique across files; letters, numbers, underscores, or hyphens |
| `depth` | Unique measured depths in metres, on a 0.125-m grid |
| `segment_id` | Continuous source-log segment; windows cannot cross a segment or a depth gap |
| `interval_id` | Recorded lithology interval, unique within the well, contiguous and carrying a single class label |
| `class_id` | Nonnegative integer nominal lithology code; numerical order has no geological meaning |
| `MSFL`, `LLS`, `LLD` | Resistivity in ohm m; finite nonmissing values must be positive |
| `DEN` | Bulk density in g/cm^3 |
| `DT` | Acoustic transit time in microseconds per foot, not microseconds per metre |
| `GR` | Natural gamma ray in API |
| `NPHI` | Dimensionless neutron-porosity fraction, not percent |

Class codes need not be identical across independently fitted wells. Their original meanings must be documented by the data owner. The implementation remaps retained codes to contiguous local class IDs starting at zero and records the map. Missing labels are not imputed. Normalize lithology names and codes before preparing the input CSV.

Curves may contain isolated NaNs; imputation medians are fitted from training windows. Entirely missing curves are rejected. Missing-value sentinel numbers such as -999 must be converted to NaN before use. Negative or zero resistivity is rejected. Automatic range-based unit conversion is not performed: the data provider must confirm units and depth alignment.

Class inclusion requires at least 40 rows and five recorded intervals per well. At least two classes must qualify. Excluded classes produce gaps, and segments are reconstructed after filtering. Candidate centers require a full nine-point window. Class-stratified target splits have nominal fractions 70/15/15 with integer counts and at least one validation/test center per retained class.

Do not fabricate interval IDs to satisfy the inclusion rule. The field `interval_id` refers to the original geological labeling intervals. It is different from a computational window or a random sample group.

The portable adapters record source file hashes, eligibility counts, class mappings, preprocessing statistics, and context overlap in the local output directory. These files may reveal restricted data; they are local run artifacts, not part of the prepared code release.
