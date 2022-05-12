## Verifier Labels 
We released additional verifier labels for Round 1,2 and 3 on May 11, 2022.  
The labels are in [`verifier_labels/verifier_labels_R1-3.jsonl`](https://github.com/facebookresearch/anli/blob/main/verifier_labels/verifier_labels_R1-3.jsonl).

## File Format
Each line in the jsonl file records the verifier labels for one example (data point) in the ANLI dataset.  
You can use the `uid` field to map the verifier labels back to the original released examples.  

Example:
```python
{"uid": "385b7051-d09f-4ecb-9224-fcc5f0615408", "verifier labels": ["e", "n", "n"]}
{"uid": "44ee99dc-4179-4160-885d-98e17f203bac", "verifier labels": ["c", "c"]}
...
```

## Verifier labels statistics
The table below shows the number of verification labels for each split in ANLI R1 to R3.  
Note that for all the examples in dev and test split, there are at least 2 verifiers that agreed with the writer of the example.  

Number of verification labels|0|2|3
---|---|---|---
R1-Train|13816|2526|604
R1-Dev|0|702|298
R1-Test|0|740|260
R2-Train | 39895| 3843| 1722
R2-Dev |            0    |   710   |  290
R2-Test |          0    |   672  |   328
R2-Train |       85751 |  10030|   4678
R2-Dev |      0    |   809   |  391
R2-Test |     0   |    820    | 380  

Total number of examples: 169265  

