# importing required libraries
from easygui import *
from tensorflow import keras
from mysql.connector import connect

db = connect(user='root',host='localhost',passwd='wrb22',database='railway')
cur = db.cursor()
#secure key for password
KEY = b'4ruL05GjqkDy_42MD02nuM_twlGYqjdqz-8PUPhisJA='

#creating the dataset 
data = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=200000)

#creating the word index
word_index = data.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3


#defining functions
def trainer(NAME,EPOCHS,BATCHSIZE,train_data,test_data):
    """This function trains the neural network and saves it as trainedmodel.h5 

    Args:
        EPOCHS ([INT]): [No of training cycle]]
        BATCHSIZE ([INT]): [number of training samples to work through before the model's internal parameters are updated
        train_data ([ITERABLE]): [training data]
        test_data ([ITERABLE]): [validation data]

    Returns:
        [None]: [nothing]
    """
    #reversing the word index
    reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])

    #padding the training and testing data
    train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding='post', maxlen=250)
    test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding='post', maxlen=250)


    '''def decode_review(text):
    
        return " ".join([reverse_word_index.get(i,'?') for i in text])'''

    #model compilation
    model = keras.Sequential()
    model.add(keras.layers.Embedding(200000,25))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(25,activation='relu'))
    model.add(keras.layers.Dense(1,activation='sigmoid'))
    model.summary()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    x_val = train_data[:10000]
    x_train = train_data[10000:]

    y_val = train_labels[:10000]
    y_train = train_labels[10000:]

    #training the model
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCHSIZE, validation_batch_size=(x_val,y_val), verbose=1)
    model.save(f"saved_model/{NAME}")


def review_encode(string:str):
    """Encodes the review string with the internal index

    Args:
        s (Str): Review string

    Returns:
        List: Encoded message
    """
    encoded = [1]
    for word in string:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded    


def testing_data(filepath=None,txt=None,Modelfile='saved_model\\my_model'):
    """Tests the data using the given inputs

    Args:
        filepath (Str, optional): file path of the text file. Defaults to None.
        txt (str, optional): review text. Defaults to None.
        Modelfile(str,optional): model file location
        Note: either of the arguments needs to be passed

    Returns:
        Tuple: returns tuple which contains score of the review
    """
    #loading the module from file
    model = keras.models.load_model(Modelfile)

    
    if filepath is not None: #if filepath is given
        with open(filepath, encoding='utf-8') as f: #opening the txt file
            for line in f.readlines():
                nline = line.replace(',','').replace('.','').replace('(','').replace(')','').replace(':','').replace('\"','').strip().split(' ')    #replacing unwanted characters
                encode = review_encode(nline)   #encoding the txt with the word index
                encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding='post', maxlen=250)
                predict = model.predict(encode)     #predicting the result
                string = f'Your Input line was:\n+{line}'
                score = f'Rating: {evaluvate(predict[0])} \n SCORE :{predict[0]}'
                textbox(msg=score,text=string)
        return 


    elif txt is not None:   #if txt is given
        nline = txt.replace(',','').replace('.','').replace('(','').replace(')','').replace(':','').replace('\"','').strip().split(' ')     #replacing unwanted characters
        encode = review_encode(nline)   #encoding the txt with the word index
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding='post', maxlen=250)
        predict = model.predict(encode) #predicting the result
        string = f'Your Input line was:\n+{txt}'
        return predict[0],string


def choice_type():
    """function for testing data by type choice


    Returns:
        function(optional): function for the next action as selected by the user
    """

    txt = textbox(msg='Please enter your review here:',text='[here]')
    if txt is None or txt == '[here]':
        if ynbox('Your Input Wasn\'t Valid Would You Like To Continue?'):
            return choice_type()
        else:
            return
    try:
        predicted = testing_data(txt=txt)
        scr = f'Rating: {evaluvate(predicted[0][0])} \n SCORE :{predicted[0][0]}'
        textbox(msg=scr,text=predicted[1])
        if ynbox(msg='Do You Want To Test More?'):
            return choice_type()
        else:
            return

    except OSError:
        modelfile = fileopenbox('Please browse and open the model')
        predicted = testing_data(txt=txt,Modelfile=modelfile)
        scr = f'Rating: {evaluvate(predicted[0][0])} \n SCORE :{predicted[0][0]}'
        textbox(msg=scr,text=predicted[1])
    except:
       
        if ynbox(msg='Your Model File Wasn\'t Found, Do You Want To Try Training A New Model ?'):
            return choice_train()
        else:
            return


def choice_from_file():
    """function for testing data by file choice

    Returns:
        function(optional): function for the next action as selected by the user
    """

    filepath = fileopenbox('Please browse the text file')
    if filepath is None:
        if ynbox('Your Input Wasn\'t Valid Would You Like To Continue?'):
            return choice_type()
        else:
            return

    try:
        testing_data(filepath=filepath)
        if ynbox(msg='Do You Want To Test More?'):
            return choice_from_file()
        else:
            return

    except OSError:
        modelfile = fileopenbox('Please browse and open the model')
        modelfile = [x for x in modelfile.split('\\')]
        modelfile = modelfile[:-1]
        modelfile = '\\'.join(modelfile)
        testing_data(filepath=filepath,Modelfile=modelfile)
        return
    except:
        if ccbox(msg='Please Try Again', cancel_choice=None) is not None:
            return choice_from_file()
        else:
            return


def choice_train():
    """ function for training the model 

    Returns:
        function(optional): function for the next action as selected by the user
    """
    fieldNames = ['Name','Epochs','Batchsize']
    data = multenterbox('Please enter the following data',fields=fieldNames)
    if data is None:
        return
    for i in data:
        if i.strip() == '':
            ccbox(f'Please Give A Valid Input For {fieldNames[data.index(i)]}')
            return choice_train()
    data = list(data)
    try:
        trainer(data[0],int(data[1]),int(data[2]),train_data,test_data)
        if ynbox('Succesfully Trained! Do You want to train more?') :
            return choice_train()
        else:
            return 
    except:
        if ynbox('Error Occured. Do You Want To Try Again?'):
            return choice_train()
        else:
            return


def evaluvate(scr):
    """returns the rating fot the given score

    Args:
        scr (int): input score

    Returns:
        str: rating
    """
    if scr < 0.3:
        return 'BAD'
    elif scr < 0.65 and scr >= 0.3:
        return 'MODERATE'
    else:
        return 'GOOD'


def login():
    fieldNames = ['Username','Password']
    user_info = multpasswordbox(msg = 'Please Login with your Username and Password',title='Login',fields=fieldNames)
    if user_info  is None:
        if ynbox('Invalid Input Do You Want To Try Again?'):
                return login()
        else:
            exit()
    user,passwd = user_info
    cur.reset() 
    cur.execute(f'select * from users WHERE Name = "{user}" and pass = "{passwd}"')
    response = list(cur)
    if len(response) > 0 :
        msgbox("Welcome")
        return main()
    else :
        if ynbox('Password does not exists try login again ?'):
            return login()
        elif ynbox('Password does not exists try registering ?'):
           return register()
        else:
            exit()

def register():
    """It is the function that registers a user

    """
    fieldNames = ['Username','Password']
    user_info = multpasswordbox(msg = 'Please Register your Username and Password',title='Register',fields=fieldNames)
    if user_info  is None:
        if ynbox('Invalid Input Do You Want To Try Again?'):
                return register()
        else:
            exit()
    user,passwd = user_info    
    print(user,passwd)
    cur.reset() 
    cur.execute(f'INSERT INTO users VALUES("{user}","{passwd}")')
    db.commit()
    ccbox(msg='Registered ! Please login with your username and password')
    return login()
#running the main loop
def main():
    """It is the main loop that runs the program
    """
    choice = buttonbox('Please Choose what do you want to do?', choices=['Train', 'Test','exit'])
    if choice == 'Test':
        choice2 = buttonbox('Please choose wether you want type or would you like to upload a text file',choices=['Type','From File','Go Back'])
        if choice2 == 'Type':
            choice_type()
            return
        elif choice2 == 'From File':
            choice_from_file()
            return
        else:
            return

    elif choice == 'Train':
        choice_train()
        return
    
    elif choice == 'exit' :
        exit(0)
    else:
        exit(0)


if __name__ == '__main__':
    login()
    while True:
        main()