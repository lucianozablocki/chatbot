import sys, pandas as pd, pickle, numpy as np, preprocesamiento
from preprocesamiento import quitarStopwordsinput,limpiarSignosinput,AutocorrectorInput,Stemmizarinput
from nltk.corpus import stopwords
import sklearn
import datetime

class Chatbot:
    def __init__(self):
        path_ans = './csv_files/respuestas - time.csv' # csv file containing the answers that the chatbot must return, ordered by node
        path_adjMat = './csv_files/adjMat_time_cambiocontexto.csv' # adjacency matrix (1 in N,M if the chatbot can navigate from N node to M node, otherwise 0)
        path_placeholders =  './csv_files/placeholders.csv' # information that can change over time, such as contact numbers, addresses
        filename = './SVC_stem_time.sav' # .sav file with trained IA model, opened with pickle
        path_bow = "./bow_w_stopwords_time.sav" # bow used to transform user input in a vector that the model can understand
        registro = "./registro.txt" # file in which the interaction is recorded

        R_pd = pd.read_csv(path_ans,delimiter=',',header=None)
        adjMat_pd = pd.read_csv(path_adjMat,delimiter=',',header=None)
        placeholders_pd = pd.read_csv(path_placeholders,delimiter=',',header=None)

        self.R = R_pd.values
        self.adjMat = adjMat_pd.values
        self.placeholders = placeholders_pd.values
        self.loaded_model = pickle.load(open(filename, 'rb'))
        self.bow_unigram = pickle.load(open(path_bow, 'rb'))

        self.thres = 0.03054759
        self.actual_node = 116

        self.stoplist = stopwords.words('spanish')
        self.keys = self.placeholders[:,0]
        self.values = self.placeholders[:,1]
        self.replacements = dict(zip(self.keys,self.values))

        self.f = open (registro, "w")

        self.user_data = {}

    def str_to_bool(self, s):
        if s == "true":
            return True
        elif s == "false":
            return False
        else:
            raise ValueError("Cannot covert {} to a bool".format(s))
    
    def preprocesar(self, sentence):
        # print(f"preprocesar method, input its: {sentence}")
        i = quitarStopwordsinput(sentence.split()) #we are using whitespaces as separator, potential problem
        # print(f"output after quitarstopwords and input to limpiarsignos: {i}")
        #num parameter in split can delimite the number of elements in the resulting list
        i = limpiarSignosinput(i)
        # print(f"output after limpiarsignos and input to autocorrector: {i.split()}")
        i = AutocorrectorInput(i.split())
        # print(f"output after autocorrector and input to stemming: {i}")
        i = Stemmizarinput(i)
        return i.strip()

    def get_node(self, case):
        switcher = {
            "bachiller" : 69,
            "licenciatura" : 107,
            "curso": 108,
            "tecnicatura": 109,
            "posgrado": 110,
            "reconquista": 88,
            "galvez": 111,
            "rafaela": 112
        }

        return switcher.get(case.lower(), 114) #default->error adm
    
    def fill_write_response(self):
        chatbot_response = self.R[self.actual_node][1]
        for placeholder in self.replacements:
            chatbot_response = chatbot_response.replace(f'<{placeholder}>', self.replacements[placeholder])
        # print(self.R[self.actual_node][1])
        self.f.write("chatbot :" + chatbot_response + "\n")
        
        if(self.actual_node==106):
            # print(datetime.datetime.now().strftime("%H:%M:%S")) 
            self.f.write(datetime.datetime.now().strftime("%H:%M:%S") + "\n")
            chatbot_response = chatbot_response + datetime.datetime.now().strftime("%H:%M:%S")
        self.f.flush()

        return chatbot_response.\
            replace("á","a").\
            replace("é","e").\
            replace("í","i").\
            replace("ó","o").\
            replace("ú","u").\
            replace("Á","A").\
            replace("É","E").\
            replace("Í","I").\
            replace("Ó","O").\
            replace("Ú","U")

    def get_response(self, input):
        if (self.str_to_bool(input["contains_variable"].lower())):
            self.f.write("user: " + input["input"] + "\n")
            self.actual_node = self.get_node(input["input"])
            self.user_data[input["variable_name"]] = input["input"] 

            chatbot_response = self.fill_write_response()
            return {
                "output": chatbot_response,
                "contains_variable": False,
                "variable_name": None
            }

        previous_answer_type = self.R[self.actual_node][2]
        self.f.write("user: " + input["input"] + "\n")
        pre_input = self.preprocesar(input["input"])

        model_input = self.bow_unigram.transform([pre_input])
        probs = self.loaded_model.predict_proba(model_input)
        print(f"predicted node is {np.argmax(probs)}")
        print(f"with probability of {max(max(probs))}")
        
        if(any(x > self.thres for x in probs[0])):
            if self.actual_node == 116:
                probs_flux = probs
            else:
                probs_flux = probs*self.adjMat[self.actual_node][0:107]
            next_node = np.argmax(probs_flux)
            self.actual_node = next_node
            if self.actual_node == 69:
                if self.user_data.get("career_type") is None: # student's career has not been loaded yet
                    chatbot_response = "Por favor, decime que carrera queres estudiar"
                    self.f.write("chatbot :" + chatbot_response + "\n")
                    self.f.flush()
                    return {
                        "output": chatbot_response,
                        "contains_variable": True,
                        "variable_name": "career_type"
                    }
                else: # student's career is known
                    self.actual_node = self.get_node(self.user_data.get("career_type"))
            elif self.actual_node == 88:
                if self.user_data.get("city") is None: # student's city has not been loadded yet
                    chatbot_response = "Por favor, decime con que sede te queres contactar"
                    self.f.write("chatbot :" + chatbot_response + "\n")
                    self.f.flush()
                    return {
                        "output": chatbot_response,
                        "contains_variable": True,
                        "variable_name": "city"
                    }
                else: # student's city is known
                    self.actual_node = self.get_node(self.user_data.get("city"))
        elif previous_answer_type == 1: #administrative error
            self.actual_node = 114
        elif previous_answer_type == 2: #academic error
            self.actual_node = 115
        else: #general error
            self.actual_node = 113
        
        print(f"actual node is: {self.actual_node}")
        chatbot_response = self.fill_write_response()

        return {
            "output": chatbot_response,
            "contains_variable": False,
            "variable_name": None
        }