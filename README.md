# Music Harmonizer - A Deep Learning Approach to Adding Chords to Melodies

## Introduction and Motivation

Melody harmonization is a large part of many composers' workflows. While melodies can stand on their own and sound beautiful, it's harmonies that give them character, context, and depth. Choosing harmonies is an artist's creative choice, however, creativity alone is not always enough. The process requires a good understanding of music theory and, often, a lot of trial and error, which can overwhelm composers, both novice and experienced. 
This opens the need for tools that can help with music harmonization.

The motivation behind creating such a tool isn't replacing composers -- harmonization is a large artistic choice and should be done by artists themselves. Rather, the tool is aimed at giving composition students and novices a helping hand in their learning process, and at helping experienced composers brainstorm and test their ideas more efficiently.

This project is done as part of the Applied Deep Learning (194.077) course at the Vienna University of Technology.

## Project Description and Approach

The goal of this project is to create a deep learning system that will be able to add chords to accompany melodic lines. For more information about the process of harmonization in general, please refer to this article: [Harmonization](https://en.wikipedia.org/wiki/Harmonization).

Many approaches to this problem generate the harmonies note-by-note, one for each voicing (for such an approach please refer to the next section of this document). This project will, on the other hand, focus on generating entire chords at once, outputting functional chord notation (e.g., I, V7, vii-7/V, etc.). This approach aims to create more harmonically coherent progressions more suitable for homophonic pieces, with the tradeoff being the loss of detail in [voice leading](https://en.wikipedia.org/wiki/Voice_leading).

Initially, transformer-based models will be explored due to their strength in handling sequential data. As part of finding an efficient architecture, other approaches will be considered, such as Long Short-Term Memory (LSTM) networks, and possibly even some reinforcement learning methods.

Type of project: **Bring your own method**.

## Previous Work - Related Scientific Papers

Arguably the most famous paper in automated music harmonization is [_DeepBach: a Steerable Model for Bach Chorales Generation_ by Gaëtan Hadjeres, François Pachet, and Frank Nielsen](https://arxiv.org/pdf/1612.01010.pdf). To train the model, they used a dataset of J.S. Bach's chorales. The model's input is one of the chorale's voices (soprano, alto, tenor, or bass), and the model returns notes of the remaining voices. Their architecture consists of two deep relational neural networks and two regular artificial neural networks, combined as shown in the figure below. The model uses music that comes both before and after the note it is generating, which helps it understand musical context better.

<p align="center">
<img src="https://github.com/user-attachments/assets/6128cb18-e257-4d43-8764-afefbf46dcc5" alt="DeepBach model architecture" width="400"/>
</p>

Another paper that deals with melody harmonization is [_Translating Melody to Chord: Structured and Flexible Harmonization of Melody With Transformer_ by Seungyeon Rhyu, Hyeonseok Choi, Sarah Kim, and Kyogu Lee](https://ieeexplore.ieee.org/abstract/document/9723052). As part of the study, three models were created: a standard transformer (STHarm), a variational transformer (VTHarm), and a regularized variational Transformer (rVTHarm). According to the study results, the models performed similarly to human composers in subjective evaluations given by 32 participants in four metrics: harmonicity (H), unexpectedness (U), complexity (C), and preference (P), as is shown in the figure below taken from the before mentioned paper. These results demostrate the usefulness of transformer-based architectures for this type of project.

<p align="center">
<img src="https://github.com/user-attachments/assets/9fdc989d-1b1e-4623-855e-fa9022f2ae2b" alt="DeepBach model architecture" width="800"/>
</p>

## Dataset

For the model training, the [_Beethoven Piano Sonata with Function Harmony (BPS-FH) dataset_](https://github.com/Tsung-Ping/functional-harmony.git) will be used. It consists of functional harmony annotations of the 1st movements from 32 Beethoven's piano sonatas. Each piece contains annotated note events (onset, note duration, measure, etc), beats, down beats, chords, and phrases. The dataset contains 86,950 note events and 7,394 chord labels in total. This dataset was chosen because of its consistency in style, with hopes of getting consistent Western classical progressions. 

In case this dataset proves not to be sufficient for training such a model, other datasets will be tested, such as the [_Jazz Audio-Aligned Harmony (JAAH) Dataset_](https://mtg.github.io/JAAH/), which contains 113 Jazz tracks with 17600 chord segments, and the [_ChoCo Dataset_](https://github.com/smashub/choco), which provides a unified dataset of over 20,000 timed chord annotations of scores, gathered from 18 different datasets.

## Work Breakdown Structure

1. Dataset preparation and preprocessing - 5 hours
    - This part includes acquiring the dataset, analyzing the data, and doing the necessary preprocessing to make it usable in deep learning models.
2. Model design and selection - 20 hours
    - This is an iterative process in which different model architectures will be tested and evaluated. After an analysis of the different designs, one of the models will be chosen for fine-tuning.
3. Model training and fine-tuning - 10 hours
    - During this step, the selected architecture will be trained on the data and the model hyperparameters will be fine-tuned. The result will be a trained model accompanied by performance metrics gathered from the test dataset.
4. Building an application to present the results - 10 hours
    - An interactive application will be created, allowing users to input their own melodies and receive the harmonized version.
5. Writing the final report - 6 hours
    - A comprehensive report of the model creation and its results will be created.
6. Preparing the presentation of the work - 4 hours
    - Creating a presentation of all the results.

# Milestone 2 Deliverables

## Error metric

Cross-Entropy Loss was used as the model's metric. It calculates the difference between the predicted probability distribution and the true class for each chord property, and sums them all up. This metric was minimized during training.

Since four loss values were summed up in each of its calculations, it would be larger than usual, which is why I aimed for a loss at around 1.

However, I didn't achieve this loss, and was only able to go down to ~4.5.

Accuracy was also taken into account, and the highest accuracy achieved on the test set was 61.3%.

## Actual Time Spent On Each Part (to the nearest 1-2 hours)

1. Dataset preparation and preprocessing - 15 hours
    - This part includes acquiring the dataset, analyzing the data, and doing the necessary preprocessing to make it usable in deep learning models.
2. Model design and selection - 20 hours
    - This is an iterative process in which different model architectures will be tested and evaluated. After an analysis of the different designs, one of the models will be chosen for fine-tuning.
3. Model training and fine-tuning - 5 hours
    - During this step, the selected architecture will be trained on the data and the model hyperparameters will be fine-tuned. The result will be a trained model accompanied by performance metrics gathered from the test dataset.
4. Building an application to present the results - ? hours
    - An interactive application will be created, allowing users to input their own melodies and receive the harmonized version.
5. Writing the final report - ? hours
    - A comprehensive report of the model creation and its results will be created.
6. Preparing the presentation of the work - ? hours
    - Creating a presentation of all the results.

