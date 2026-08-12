# Dataset Notes

Raw MRI images are intentionally not included in this public GitHub-ready package.

The submitted thesis archive contained:

| Dataset | Classes / Source | Images |
|---|---|---:|
| Main training/evaluation dataset | Brain Tumor | 2,513 |
| Main training/evaluation dataset | Healthy | 2,087 |
| Separate unseen test set | Tumor | 300 |
| Separate unseen test set | Healthy | 300 |
| Hospital evaluation set | Buner Hospital data | 73 |

## Expected Local Structure

```text
data/
├── main/
│   ├── Brain Tumor/
│   └── Healthy/
├── unseen/
│   ├── (test)Tumor/
│   └── (test)Healty/
└── hospital/
```

### Public-repository note

The Buner Hospital images are excluded because clinical/hospital data should only be redistributed when the researcher has explicit authorization, appropriate consent/ethics clearance, and permission for public release.

The other MRI datasets are also omitted from this cleaned package because their redistribution rights/license were not established from the submitted ZIP. Add them locally only if their license permits redistribution.
