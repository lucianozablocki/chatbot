import torch as tr
#import numpy as np
import pandas
import preprocesamiento
#CHAR_CHANNELS = 55

class CharTokenizer():
    def __init__(self):
        #emb = np.eye(80, 55)
        #emb[54:80] = emb[28:54]
        #emb[28:54, 54] = 1
        emb = tr.eye(80,80)
        self.embedding = emb

    def tokenize(self, comments): #comments es lista de palabras
        return list(map(lambda x:self.str2tns(x), comments)) #x->a cada elemento de comments
                                                            #(palabra) le aplico str2tns

    def str2tns(self, comm): #comm es una palabra
        return tr.LongTensor(list(map(self.char2idx, comm))) #a cada elemento de la palabra 
                                                #(caracter) le aplico char2idx

    def char2idx(self, c): #c es un caracter
        idx = 27     
        #print(c)
        if c in self.char_set:
            idx = self.char_set[c]
        return idx
    
    #def get_tensor(self,idx):
    #    return list(map(lambda x: self.embedding[x],idx))

    char_set = {
            ' ' : 0, '\n': 1, '!' :  2, '"' :  3, '#' :  4, '$' :  5, '%' :  6, '&' :  7, "'" :  8,
            '(' :  9, ')' : 10, '*' : 11, '+' : 12, ',' : 13, '-' : 14, '.' : 15, '/' : 16, ':' : 17,
            ';' : 18, '=' : 19, '?' : 20, '_' : 21, '\xad' : 22, '’' : 23, '“' : 24, '”' : 25,
            '0' : 26, '1' : 26, '2' : 26, '3' : 26, '4' : 26, '5' : 26, '6' : 26, '7' : 26, '8' : 26,
            '9' : 26,
            'A' : 28, 'B' : 29, 'C' : 30, 'D' : 31, 'E' : 32, 'F' : 33, 'G' : 34, 'H' : 35, 'I' : 36,
            'J' : 37, 'K' : 38, 'L' : 39, 'M' : 40, 'N' : 41, 'O' : 42, 'P' : 43, 'Q' : 44, 'R' : 45,
            'S' : 46, 'T' : 47, 'U' : 48, 'V' : 49, 'W' : 50, 'X' : 51, 'Y' : 52, 'Z' : 53,
            'a' : 54, 'b' : 55, 'c' : 56, 'd' : 57, 'e' : 58, 'f' : 59, 'g' : 60, 'h' : 61, 'i' : 62,
            'j' : 63, 'k' : 64, 'l' : 65, 'm' : 66, 'n' : 67, 'o' : 68, 'p' : 69, 'q' : 70, 'r' : 71,
            's' : 72, 't' : 73, 'u' : 74, 'v' : 75, 'w' : 76, 'x' : 77, 'y' : 78, 'z' : 79}
    """
    char_set={'a':0,'b':1,'c':2}
    """

if __name__=='__main__':
    dataset = pandas.read_csv("pregTest.csv",header=None,delimiter=',')
    #x_text = dataset.values[:,1]
    char_lvl = CharTokenizer()
    x = dataset.values
    x_text = preprocesamiento.preprocesar(x,1)
    x_text = x_text[:,1]
    for sentence in x_text:
        idx = char_lvl.tokenize(sentence)
        print(idx)
        print(len(idx))