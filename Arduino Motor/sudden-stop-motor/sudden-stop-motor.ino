int enA = 9;
int in1 = 8;
int in2 = 7;
int num;
int val = 0;

void setup() {
  // put your setup code here, to run once:
  pinMode(enA, OUTPUT);

  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);

  
  // Turn off motors - Initial state
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  analogWrite(enA, 255);
  Serial.begin(115200);
  Serial.setTimeout(1);
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, val);

}

void loop() {
 
while (!Serial.available());
  num = Serial.readString().toInt();
  if (num == 1){
    val = 1;
    digitalWrite(LED_BUILTIN,val);
   analogWrite(enA, 255);
  digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);
  }
  else if (num == 2){
    val = 0;
    digitalWrite(LED_BUILTIN,val);
    analogWrite(enA, 175);
    digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);
  }
  else if (num == 3){
    val = !val;
    digitalWrite(LED_BUILTIN,val);
    delay(250);
    val = !val;
    digitalWrite(LED_BUILTIN,val);
    delay(250);
    val = !val;
    digitalWrite(LED_BUILTIN,val);
    delay(250);
    val = !val;
    digitalWrite(LED_BUILTIN,val);
    delay(250);
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
  }
  else{
    digitalWrite(LED_BUILTIN,0);
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
  }
}
