# Music harmonizer

## Introduction and motivation

Melody harmonization is a large part of many composers' workflows. While melodies can stand on their own and sound beautiful, it's harmonies that give them character, context, and depth. Choosing harmonies is an artist's creative choice, however, creativity alone is not always enough. The process requires a good understanding of music theory and, often, a lot of trial and error, which can overwhelm composers, both novice and experienced. 
This opens the need for tools that can help with music harmonization.

The motivation behind creating such a tool isn't replacing composers -- harmonization is a large artistic choice and should be done by artists themselves. Rather, the tool is aimed at giving composition students and novices a helping hand in their learning process, and at helping experienced composers brainstorm and test their ideas more quickly.

This project is done as part of the Applied Deep Learning (194.077) course at the Vienna University of Technology.

## Project description and approach

The goal of this project is to create a deep learning system that will be able to add chords to accompany melodic lines. For more information about the process of harmonization in general, please refer to this article: [Harmonization](https://en.wikipedia.org/wiki/Harmonization).

Many approaches to this problem generate the harmonies note-by-note, one for each voicing (for the current approaches refer to the next section of this document). This project will, on the other hand, focus on generating entire chords at once, outputting functional chord notation (e.g., I, V7, vii-7/V, etc.). This approach aims to create more harmonically coherent progressions more suitable for homophonic pieces, with the tradeoff being the loss of detail in voice leading. 

Type of project: **Bring your own method**.

## Previous work - related scientific papers

Arguably the most famous paper in automated music harmonization is [_DeepBach: a Steerable Model for Bach Chorales Generation_ by Gaëtan Hadjeres, François Pachet, and Frank Nielsen](https://arxiv.org/pdf/1612.01010.pdf). To train the model, they used a dataset of J.S. Bach's chorales. Their input was the soprano voice of the chorale (the highest voice), and the model returned notes of the remaining three voices (alto, tenor, and bass). Their model consists of two deep relational neural networks and two regular artificial neural networks, combined into an architecture shown in the figure below. The model used music that came both before and after the note it was generating, which helped it understand musical context better.

<p align="center">
<img src="https://github.com/user-attachments/assets/6128cb18-e257-4d43-8764-afefbf46dcc5" alt="DeepBach model architecture" width="400"/>
</p>

Another paper that deals with melody harmonization is [_Translating Melody to Chord: Structured and Flexible Harmonization of Melody With Transformer_ by Seungyeon Rhyu, Hyeonseok Choi, Sarah Kim, and Kyogu Lee](https://ieeexplore.ieee.org/abstract/document/9723052). As part of the study, three models were created: a standard transformer, a variational transformer, and a regularized variational Transformer. According to the study results, the models performed similar to humans in subjective evaluations given by 32 participants in four metrics: harmonicity, unexpectedness, complexity, and preference.

## Dataset

For the model training, the [_Beethoven Piano Sonata with Function Harmony (BPS-FH) dataset_](https://github.com/Tsung-Ping/functional-harmony.git) will be used. It consists of functional harmony annotations of the 1st movements from Beethoven's 32 piano sonatas. Each piece contains annotated note events (onset, note duration, measure, etc), beats, down beats, chords, and phrases. The dataset contains 86,950 note events and 7,394 chord labels. This dataset was chosen because of its consistency in style, with hopes of getting consistent Western classical progressions. 

In case this dataset proves not to be sufficient for training such a model, other datasets will be tested, such as the [_Jazz Audio-Aligned Harmony (JAAH) Dataset_](https://mtg.github.io/JAAH/), which contains 113 Jazz tracks with 17600 chord segments, and the [_ChoCo Dataset_](https://github.com/smashub/choco), which provides a unified dataset of over 20,000 timed chord annotations of scores, gathered from 18 different datasets.

## Work breakdown structure

1. Dataset preparation an preprocessing - 5 hours
    - This part includes acquiring the dataset, analyzing the data, and doing the necessary preprocessing to make it usable in deep learning models.
2. Model design and selection - 20 hours
    - This is an iterative process in which different model architectures will be tested and evaluated. After an analysis of the different designs one of the models will be chosen for fine-tuning.
3. Model training and fine-tuning - 10 hours
    - During this step, the selected architecture will be trained on the data and the model hyperparameters will be fine-tuned. The result will be a trained model acompanied by performance metrics gathered from the test dataset.
4. Building an application to present the results - 10 hours
    - An interactive application will be created, allowing users to input their own melodies and receive the harmonized version.
5. Writing the final report - 6 hours
    - A comprehensive report of the model creation and its results will be created.
6. Preparing the presentation of the work - 4 hours
    - Creating a presentation of all the results.
