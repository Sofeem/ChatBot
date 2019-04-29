# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 18:56:00 2019

@author: Sofee
"""

#import PySimpleGUI as sg
from sklearn.externals import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
import re 
#import string 
import random

stopwords_english = stopwords.words('english')

#dialogue_act_utterence = []

GREETING_RESPONSES = ["hi", "hey", "hi there", "hello", "I am glad! You are talking to me"]
BYE_RESPONSES = ["hi", "hey", "hi there", "hello", "I am glad! You are talking to me"]

compiled_whquestion =[(re.compile(x[0]), x[1]) for x in  [
                       ["what (.*) assignments (.*)",
                       ["The assigments topic will be disscused in the class",
                        "there are three assignments and every assignemnts are on different topics. the topic will be known in the class" ]],
                       ["How (.*) assignments graded",
                        ['Each assignmnets carries equal points']],
                         ["what (.*) project (.*)",
                       ["The projects topic will be disscused in the class",
                        "there are plenty of projects. the project topic will be known in the class" ]],
                       ["How (.*) projects (*) graded",
                        ['Each assignmnets carries equal points']],
                         ["what (.*) cousre content (.*)",
                       ["Foundation of text retrieval systems, Lexical ontologies, word sense disambiguation, Text categorization, Corpus-based inferences and Natural Language Processing tools. NLP The sub-domain of artificial intelligence concerned with the task of developing programs possessing some",
                        "Natural Language Processing tools. NLP The sub-domain of artificial intelligence concerned with the task of developing programs possessing some capability of ‘understanding’ a natural language in order to achieve some specific goal. The sub-domain of artificial intelligence concerned with the task of developing programs possessing some capability of ‘understanding’ a natural language in order to achieve some specific goal." ]],
                       ["when (.*) assignments (*) grded ",
                        ['After one week of assignment submission']],
                         ["what (.*)  (.*) beside studies",
                       ["Besides the study plan which is a very serious and students tend to study a lot, get good grades , and plan for the future research, internships, or projects, there are a ton of activities that students are doing during each semester. Every year there are large events that attract a huge portion of people. For example, Ilosaarirock, Joensuu Music Winter, Festival of Visual Culture Viscult, and Gospel festivals. There are excellent opportunities for sport in. For instance volleyball, basketball, football, swimming, fishing, horse riding, gyms, skiing and skating, bowling, tennis, squash, badminton, indoor climbing, table tennis, minigolf, billiards and so on.",
                        "there are three assignments and every assignemnts are on different topics. the topic will be known in the class" ]],
                       ["How (.*) group",
                        ['maximum four members can be in the group']],
                         ["what (.*) assignments (.*)",
                       ["The assigments topic will be disscused in the class",
                        "there are three assignments and every assignemnts are on different topics. the topic will be known in the class" ]],
                       ["How (.*) assignments graded",
                        ['Each assignmnets carries equal points']]
                        
 ]]
                      
compiled_statement =[(re.compile(x[0]), x[1]) for x in  [
                       ["(.*)",
                       ["Please let me know how can i help you further ",
                        "I guess i have answered all your queries",
                        "Better luck next time "]]
                       
                        
                        
 ]]                      
   

compiled_ynquestion =[(re.compile(x[0]), x[1]) for x in  [
                       ["Do (.*)",
                       ["The assigments topic will be disscused in the class",
                        "there are three assignments and every assignemnts are on different topics. the topic will be known in the class" ]],
                       ["How (.*) graded",
                        ['Each assignmnets carries equal points']]
                        
 ]]   
                       
compiled_accept =[(re.compile(x[0]), x[1]) for x in  [
                       ["(.*)",
                       ["Please let me know how can i help you further ",
                        "I guess i have answered all your queries",
                        "Better luck next time "]]
                        
 ]]                         
                       


# feature extractor function
def bag_of_words(words):
    words_clean = []
    
    for word in words:
#        print(word)
        word = word.lower()
        words_clean.append(word)
        #if word not in string.punctuation:
          
    words_dictionary = dict([word, True] for word in words_clean)  
    
    
    return words_dictionary

def respond_greet() :
     resp = random.choice(GREETING_RESPONSES)
     return resp
def respond_bye() :
    return "I guess it's time for me to go then."


def respond_question(uu) :
   
       for rule, value in compiled_whquestion:
           match = rule.search(uu)
#           print(uu)
           if match is not None:
            # found a match ... stuff with corresponding value
            # chosen randomly from among the available options
            resp = random.choice(value)
            
#            print(resp)
#    if da == 'ynquestion':
#         return "I wish I knew."
            return resp
        
#def respond_other() :
#    return ":P  Well, what next?"

def respond_statement(uu) :
#    if da == 'whquestion' :
        
       for rule, value in compiled_whquestion:
           match = rule.search(uu)
#           print(uu)
           if match is not None:
            # found a match ... stuff with corresponding value
            # chosen randomly from among the available options
            resp = random.choice(value)
            
            return resp
def respond_accept(uu) :

     for rule, value in compiled_accept:
         match = rule.search(uu)
#           print(uu)
         if match is not None:
            # found a match ... stuff with corresponding value
            # chosen randomly from among the available options
            resp = random.choice(value)
            
            return resp
    

def respond_ynQuestion(uu) :
    for rule, value in compiled_ynquestion:
         match = rule.search(uu)
#           print(uu)
         if match is not None:
            # found a match ... stuff with corresponding value
            # chosen randomly from among the available options
            resp = random.choice(value)
            
            return resp


    

def IdDA(user_utterance):
    
    utter_tokens = word_tokenize(user_utterance)
#    print(utter_tokens)
    utter_set = bag_of_words(utter_tokens)
#    print(utter_set)
    loaded_model = joblib.load('C:/Users/Sofee/Desktop/Final_Chatbot/model.pkl')
    dialogue_act = loaded_model.classify(utter_set)
    #dialogue_act_utterence.append(dialogue_act)
#    print(dialogue_act)
    
    return dialogue_act
   
    
flag=True
print("I am ChatBot: I am here to help you on your Course related and Univeristy related Materials")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    dialogue_act_utterence = IdDA(user_response)
    print(dialogue_act_utterence)
    
    if(dialogue_act_utterence == 'Greet'):
         robo_response = respond_greet()
         print(robo_response)
    if(dialogue_act_utterence == 'Bye'):
         robo_response = respond_greet()
         print(robo_response)
         flag==False
    if(dialogue_act_utterence == 'whQuestion'):
         robo_response = respond_question(user_response)
         print(robo_response)
    if(dialogue_act_utterence == 'Accept'):
         robo_response = respond_accept(user_response)
         print(robo_response)    
    if(dialogue_act_utterence == 'ynQuestion'):
         robo_response = respond_ynQuestion()
         print(robo_response)
    if(dialogue_act_utterence == 'Statement'):
         robo_response = respond_statement()
         flag==False 
         print(robo_response)
