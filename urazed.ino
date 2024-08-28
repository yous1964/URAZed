#include <stdlib.h>

// Motor Connections (ENA & ENB must use PWM pins)
#define IN1 9
#define IN2 8
#define IN3 7
#define IN4 6
#define ENA 10
#define ENB 5
#define leftBtn 2
#define rightBtn 3
#define LAnlg A0
#define RAnlg A1
#define buzzer 11

int b1;
int b2;
int xValueL;
int xValueR;
bool inWarn = 0;
char serial;

char buf[80];


int readline(int readch, char *buffer, int len) {
    static int pos = 0;
    int rpos;

    if (readch > 0) {
        switch (readch) {
            case '\r': // Ignore CR
                break;
            case '\n': // Return on new-line
                rpos = pos;
                pos = 0;  // Reset position index ready for next time
                return rpos;
            default:
                if (pos < len-1) {
                    buffer[pos++] = readch;
                    buffer[pos] = 0;
                }
        }
    }
    return 0;
}

void setup() {

  // Set motor connections as outputs'
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);

  // Start with motors off
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);

  analogWrite(ENA, 0);
  analogWrite(ENB, 0);

  Serial.begin(9600);

  pinMode(buzzer, OUTPUT);

}

void loop() {

  if(Serial.available() > 0)  {
      serial = Serial.read();
      Serial.println( serial, HEX);
      if (serial=='s'){
        inWarn = 1;
        
      }else if (serial == 'q'){
        inWarn = 0;
       }
   } 

  if (inWarn == 1){
    tone(buzzer, 1000);
  }else{
    noTone(buzzer);
  };

  b1 = digitalRead(4);
  b2 = digitalRead(3);
  xValueL = analogRead(LAnlg);
  xValueR = analogRead(RAnlg);



  if(xValueR < 460){
    backA();
  }else if (xValueR > 564){
    frontA();
  }else{
    stopA();
  };
  
  if(xValueL < 460){
    backB();
  }else if(xValueL > 564){
    frontB();
  }else{
    stopB();
  };

}


void frontA(){
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
};

void backA(){
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
};

void stopA(){
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
};

void frontB(){
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
};

void backB(){
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
};

void stopB(){
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
};

void front(){
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
};

void back(){
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
};

void stop(){
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
};

void pivotL(){
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
};

void pivotR(){
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
};
