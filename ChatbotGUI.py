# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 18:56:00 2019

@author: Sofee
"""

import PySimpleGUI as sg
from sklearn.externals import joblib
from dialogueactprediction import Classifier


GREETING_RESPONSES = ["hi", "hey", "hi there", "hello", "I am glad! You are talking to me"]
BYE_RESPONSES = ["hi", "hey", "hi there", "hello", "I am glad! You are talking to me"]
#frame_layout = [
#                  [sg.Text(''), sg.InputText()],      
#                  [sg.Submit()],      
#              ] 
#   
#
#layout = [  [sg.Output(size=(70,40)),sg.Output(size=(80,10))],
#            [sg.Frame('Meessage', frame_layout, font='Any 12', title_color='blue')],
#            [sg.InputText('Default text')],
#           
#         ]
#
#window = sg.Window('My first GUI').Layout(layout)
#window.Read()
#button, (name,) = window.Read()

#model = joblib.load('finalized_model.sav')
#question = input()
#model.predict([question])[0]

def respond_greet() :
     resp = random.choice(GREETING_RESPONSES)
     return resp

def respond_question(text, valence) :
    if text == 'whquestion' :
        return "I wish I knew."
    
def respond_other(text, valence) :
    return ":P  Well, what next?"

def respond_statement(text, valence) :
    if text == 'statement' :
        return "I wish I knew."
   
    
def respond_bye(text, valence) :
    return "I guess it's time for me to go then."


def respond_reject(text, valence) :
    if text == 'whquestion' :
        return "I wish I knew."
    else :
        return "I still think you should reconsider."
    
def respond_emphasis(text, valence) :
    if text == 'whquestion' :
        return "I wish I knew."
    else :
        return ":("

def response(user_utterance):
    Dialogue_act = IdDA(user_utterance)
    robo_response='hi'
    print (Dialogue_act)
#    time.sleep(5)
#    if(Dialogue_act == 'Greet'):
#        robo_response=respond_greet()
    return robo_response
    

def IdDA(user_utterance):
    loaded_model = joblib.load('finalized_model.sav')
    dialogue_act = loaded_model(user_utterance)
    
    return dialogue_act
    
    
    
    
    
# Blocking window that doesn't close      
def ChatBot():      
    layout = [[(sg.Text('This is where standard out is being routed', size=[40, 1]))],      
              [sg.Output(size=(80, 20))],      
              [sg.Multiline(size=(70, 5), enter_submits=True)],      
              [sg.RButton('SEND', button_color=(sg.YELLOWS[0], sg.BLUES[0]))],      
             ]

    window = sg.Window('Chat Window', default_element_size=(30, 2)).Layout(layout)

    # ---===--- Loop taking in user input and using it to query HowDoI web oracle --- #      
    while True:      
        event, value = window.Read()      
        if event == 'SEND':   
            print(value)
            chat_response  = response(event)
            print(chat_response)
        else:      
            window.AutoClose(True)

ChatBot()