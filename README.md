# creativeAI
Master's Degree Thesis, translate music language in the visive art language.
Build up a complex DNN to generate images from music streaming, describing the music path trough the tempo.

Music and images do not belong to the same domain. 
- Music domain : D(music) = M
- Image domain : D(image = I

Offline work:
Build up a Map<Emotion, List<RelatedColors>> so that we give just one color to the song (the predominant) , and we provide related colors for it 
  build whatever in order to adjust the sensibility at run-time
  

Music processing pipeline:

*.wavFormat : piece -> Spectral Analysis -> get Features about pitch, rhythm and timbre and save statistics' bins

Dataset is: music, emotions[], predominant[]  // emotions and predominant values refere to a 15s bin? 

Aim is:
  Construct tempo/features grapichs and use as histograms to attribute a pair with a emotions thus colors ONE for each feature
    i(timbre) + j(rhythm) + k(pitch) = (-)Grad(emotions)
    
    a(feature1) + b(feature2) + c(feature3) + .. + n(featureN) = 1; //NORMALIZATION /STANDARDIZATION, if = 100 will I have more sensitivy?


## Step 1
- Cercare Dataset di immagini e musica che possibilmente abbiano degli attributi shared
- WikiArt?
- ArtEmis: Affective Languege for Visual Art https://www.artemisdataset.org

## Step 2
- Creare un classificatore che classifichi le immagini con label pertinenti. (multi-class labels)
- The Classifier C : learns from piece.spectralAnalysis with piece labeled each bin and **classifies piece.Emotions based on piece's spectral analysis**
- The proposed one learns from 'noise' which is an internal representation of the values of the generator
-   if we have the spectral analaysis values of the music bin, we can attribute an emotion to a spectral analysis features set [and depending from the others]
-   so the classifier has to predict an emotion, based on a particolar SA
-     if we bring with us every BIN.SA.FeatureHistogram() knowledge we can start to make comparisons (important in Music domain) during time to compute more accurated chioces based on pattern recognition between SA mapped into emotions. (RNN / Transfomer)

  Now that we have an run-time emotion predicted with another method, we can build a CGAN/DCGAN that (if probabilistic computation tractable)
-     wants to learn emotion from artImages (see Dataset ArtEmis: artImage, emotion)
-     wants to generate artImages conditioned on emotion predicted by the RNN/T 



## Step 3 
- Aggiungere tag sul dinamismo, legato al ritmo, cosa che nelle immagini non si trova
