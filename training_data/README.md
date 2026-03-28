# Training Data

Place your DSTT training data files here. Three formats are supported:

## JSONL (recommended)

One JSON object per line with `input` and `modality` fields:

```jsonl
{"input": "A sunset over mountains", "modality": "Image"}
{"input": "Breaking news about science", "modality": "Text"}
{"input": "Time-lapse of a flower", "modality": "Video"}
```

## CSV

Standard CSV with header row:

```csv
input,modality
A sunset over mountains,Image
Breaking news about science,Text
```

## Plain Text

One input per line. All examples default to `Text` modality. Lines starting with `#` are skipped.

```text
A sunset over mountains
Breaking news about science
Time-lapse of a flower
```

## Modality Values

| Value   | Description |
|---------|-------------|
| `Text`  | Text generation (articles, stories, documentation) |
| `Image` | Image generation (photos, paintings, illustrations) |
| `Video` | Video generation (footage, animation, time-lapse) |
