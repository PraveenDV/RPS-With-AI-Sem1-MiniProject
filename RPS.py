import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mediapipe import solutions as mp
import random
import time

class rps():
    def __init__(self):
        self.model = load_model("venv\RPSModel.h5")
        self.class_names = ["paper", "rock", "scissors"]
        self.mp_hands = mp.hands
        self.mp_drawing = mp.drawing_utils
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.gamestate=True
        self.cap = cv2.VideoCapture(0)
        self.player_score=0
        self.comp_score=0
        self.nrounds=5
        self.timeframe=6
        self.prev_round_time=0
        self.curr_round=1

    def score(self, label, frame):
        display_text=''
        comp_text=''
        if self.gamestate and time.time()-self.prev_round_time>self.timeframe:
            if label:
                self.prev_round_time=time.time()
                comp_choice=random.choice(self.class_names)
                if label==comp_choice:
                    display_text='Tie!'
                    #comp_text=comp_choice
                    pass
                if label=='rock' and comp_choice=='scissors' or label=='paper' and comp_choice=='rock' \
                      or label=='scissors' and comp_choice=='paper':
                    self.player_score+=1
                    display_text='Player wins!'
                    #comp_text=comp_choice
                else:
                    self.comp_score+=1
                    display_text='Computer wins!'
                    #comp_text=comp_choice
                self.curr_round+=1
                comp_text=comp_choice

        if time.time()-self.prev_round_time>1:
            display_text='Progressing to next round'
           # self.gamestate=False

        cv2.putText(frame, f'Computer Score: {self.comp_score}  Player Score: {self.player_score}', (10,30),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, display_text, (10,70),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, f'Computer choice: {comp_text}', (10,90),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        #cv2.putText(frame, f'Time elapsed: {time.time()-self.prev_round_time}', (300,40),
         #                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                
        if self.curr_round>=self.nrounds:
            self.gamestate=False
            if self.player_score>self.comp_score:
                cv2.putText(frame, 'Player wins the game!', (10,110),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 3)
            if self.player_score<self.comp_score:
                cv2.putText(frame, 'Computer wins the game!', (10,110),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 3)
            
            #self.release()
    def main(self):
        while True:
            ret, frame = self.cap.read()
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb)

            if result.multi_hand_landmarks:
                for lm in result.multi_hand_landmarks:
                    xl = [int(p.x * w) for p in lm.landmark]
                    yl = [int(p.y * h) for p in lm.landmark]

                    x_min, x_max = max(min(xl)-20,0), min(max(xl)+20,w)
                    y_min, y_max = max(min(yl)-20,0), min(max(yl)+20,h)

                    roi = frame[y_min:y_max, x_min:x_max]

                    try:
                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        resized = cv2.resize(gray, (180, 180))
                        normalized = resized / 255.0
                        reshaped = np.reshape(normalized, (1, 180, 180, 1))

                        predict = self.model.predict(reshaped)
                        label_idx = np.argmax(predict, axis=1)[0]
                        label=self.class_names[label_idx]
                                                
                    except Exception as e:
                         print("Error:", e)
                         continue
                    
                    self.mp_drawing.draw_landmarks(frame,lm,self.mp_hands.HAND_CONNECTIONS)
                    try:
                        self.score(label, frame)
                    except Exception as e:
                        print("Scoring Error:", e)
                    cv2.putText(frame, label, (x_min, y_min-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)


            cv2.imshow("frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__=="__main__":
    game = rps()
    game.main()
    #if game.gamestate=='end':
    game.release()




        
'''def result(self):
            idx=random.randint(0,2)
            current=time.time()
            for i in range(self.nrounds):
             if label and self.prev_round_time-current < self.timeframe:
                self.gamestate='play'
                comp_choice=self.class_names[idx]
                if label=='rock':
                        time.sleep(2)
                        if comp_choice=='rock':
                            print('Tie!')
                        if comp_choice=='paper':
                            print('Computer wins!')
                            self.comp_score+=1
                        if comp_choice=='scissors':
                            print('Player wins!')
                            self.player_score+=1
                elif label=='paper':
                        self.prev_round_time=current
                        if comp_choice=='rock':
                            print('Player wins!')
                            self.player_score+=1
                        if comp_choice=='paper':
                                print('Tie!')
                        if comp_choice=='scissors':
                                print('Computer wins!')
                                self.comp_score+=1
                elif label=='scissors':
                        self.prev_round_time=current
                        if comp_choice=='rock':
                            print('Computer wins!')
                            self.comp_score+=1
                        if comp_choice=='paper':
                                print('Player wins!')
                                self.player_score+=1
                        if comp_choice=='scissors':
                                print('Tie!')
                cv2.putText(frame, f'Computer Score: {self.comp_score}  Player Score: {self.player_score}', (10,30),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            self.gamestate='end'''