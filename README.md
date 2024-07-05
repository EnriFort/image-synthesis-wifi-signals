# image-synthesis-wifi-signals
1) The STRUCTURE of the code is as follows:
    - **data** - contains datasets and other useful data:
        - **OBJECTS**: dataset for objects:
            - **img**: image dataset;
            - **sgnl**: signal dataset, amplitudes over time (only signlas with 50 pkts are available on the repository);
            - **models**: where to save best models;
            - **results**: where to save final generated images.

    - **Network** - Folder with the actual code
        - **customDataset.py**: custom dataset function to open images and signals;
        - **customDatasetPol.py**: modified version of 'customDataset.py' to combine different polarizations;
        - **main.py**: main function where the training and instantiation of the neural network take place;
        - **networks**: contains the implementation of the neural network: encoder, decoder, and LSTM;
        - **trainSteps**: contains functions for the training and evaluation phases of the network: train_loop, validation_loop, and test_loop;
        - **utils.py**: contains useful functions called by the main and other scripts.
      
2) To perform various TESTS, you can pass the Hyperparameters (*learning rate, batch size...*) from the command line; just run the python script 'main.py' with the various parameters, for example:

        `python .\Network\main.py --img_size=128 --lr=0.0002 --batch_size=16 --wd=...`

If no parameters are passed, the default ones will be used. For more information about the existing parameters, read the main.

Above there is the proposed model architecture for Image synthesis from Wi-Fi signals.
![modello](https://github.com/EnriFort/image-synthesis-wifi-signals/assets/50843864/efcdcc52-5e5f-466e-9ed0-ac2091109cbe)
