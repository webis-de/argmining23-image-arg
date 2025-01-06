# ImageArg 2023 Submission - Team "feeds"

## Abstract
This repository contains the submissions and analysis of Team "feeds" for ImageArg 2023. Our team participated in two subtasks: Argumentative Stance Classification (Subtask A) and Image Persuasiveness Classification (Subtask B).

### Subtask A: Argumentative Stance Classification
In this subtask, we achieved an F1 score of 0.84, showcasing a strong performance in classifying the stance based on tweet text. However, our analysis revealed challenges in classifying straightforward sentences. We also discussed the potential for future work involving the incorporation of image data.

### Subtask B: Image Persuasiveness Classification
Our approach for Subtask B secured the first position with an F1 score of 0.56. While we effectively classified images that did not enhance persuasiveness, we encountered difficulties in identifying images that enhanced the text's persuasiveness. We emphasized the need for advanced feature engineering to improve the model's ability to identify nuanced persuasive elements within images. Additionally, we found that our classifiers performed nearly as well without considering the text, highlighting the influential role of CLIP image embeddings in the decision-making process.

## Repository Contents
- `code/`: Contains the code used for our submissions.
- `results/`: Results and analysis of our submissions.
- `README.md`: This readme file summarizing our participation.

Please note that the dataset used for this competition is not included in this repository. You can obtain the dataset from https://imagearg.github.io/

## Conclusion
Our participation in ImageArg 2023 provided valuable insights into the challenges and opportunities in multi-modal analysis, combining visual and textual information for argumentative analysis. For more detailed information, please refer to our paper:

[Link to Paper](https://aclanthology.org/2023.argmining-1.16/)

For inquiries or collaborations, please feel free to reach out to our team.

We look forward to potential future research directions in this field.

Best regards,
Team "feeds

