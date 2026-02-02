# Evaluation Results Table

| Model Name | Top-1 Accuracy (%) | Top-5 Accuracy (%) | 
|------------|-------------------|-------------------|
|small        | 84.278%           | 97.048% | 
|torchao, attn only, 8 bit | 85.72% | 97.62% | 
|fixed attn only 8 bit, 4 fl | 85.61$ | 97.59% |
|fixed attn only 2 bit, 2 fl | 3.48% | 9.19% |
|fixed attn only 4 bit, 2 fl | 81.86% | 96.07% | 
|fixed all, 8 bit, 4 fl| 82.84% | 97.46% |
|fixed all, 4 bit, 2 fl|  0.61% | 2.73%
|fixed all 2 bit, 1 fl | 0.1% | 0.53% |


| Model Name | Top-1 Accuracy (%) | Top-5 Accuracy (%) | 
|------------|-------------------|-------------------|
|tiny        |    80.64%     | 95.51% | 
|fixed all, 8 bit, 4 fl| 77.1% | 95.2% |
|fixed all, 4 bit, 2 fl| ~0% | ~0%
|fixed all 2 bit, 1 fl | - |- |

Fixed operations
basic tiny

| Model Name | Top-1 Accuracy (%) | Top-5 Accuracy (%) | 
|------------|-------------------|-------------------|
|fixed all, 16 bit, 8 fl| 80% | 95.2% |
|fixed all, 12 bit, 6 fl| 53.4% | 78.3% |
|fixed all, 8 bit, 4 fl| 0.14% | 0.61% |


