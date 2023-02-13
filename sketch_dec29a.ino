
#include "AESLib.h"
#include <Arduino.h>
#include <time.h>
//#include <cstdio> 
//#include <cstdlib>
#include <stdio.h>

#define BAUD 9600

AESLib aesLib;


#define INPUT_BUFFER_LIMIT 128


char plaintext[INPUT_BUFFER_LIMIT] = {0};  // THIS IS INPUT BUFFER (FOR TEXT)
char ciphertext[2*INPUT_BUFFER_LIMIT] = {0}; // THIS IS OUTPUT BUFFER (FOR BASE64-ENCODED ENCRYPTED DATA)

// AES Encryption Key (same as in node-js example)
// key generation
byte* aes_key = new byte[16];

// General initialization vector (same as in node-js example) (you must use your own IV's in production for full security!!!)
//iv generation
byte aes_iv[N_BLOCK] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

// dec to hex
/*char* DecIntToHexStr(long long num){
  char* str;
  long long temp = num/16;
  int left = num % 16;
  if (temp > 0){
    str += DecIntToHexStr(temp);
  }
  if (left<10){
    str += (left + '0');
  }
  else{
    str += ('A' + left - 10);
  }
  return str;
}*/
char* randstr(char str[], int n){
  int i, randno;
  time_t t;
  srand((unsigned)time(&t));
  char str_array[63] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  for (i=0; i<n; i++){
    randno = random()%62;
    str[i] = str_array[randno];   
  }
  str[n] = '\0';
  return str;
}
void aes_init() {
  aesLib.gen_iv(aes_iv);
  // key randommize
  //Serial.print("Key is: ");
  for(int i =0; i<16; i++){
    char str[100];
    char str1[100];
    itoa(random()%16, str, 16);
    itoa(random()%16, str1, 16);
    strcat(str, str1);
    Serial.print(str);
    aes_key[i] = str;
  }
  Serial.println("");

  aesLib.set_paddingmode((paddingMode)0);
}

uint16_t encrypt_to_ciphertext(char * msg, byte iv[]) {
  int msgLen = strlen(msg);
  int cipherlength = aesLib.get_cipher64_length(msgLen);
  char encrypted_bytes[cipherlength];
  uint16_t enc_length = aesLib.encrypt64((byte*)msg, msgLen, encrypted_bytes, aes_key, sizeof(aes_key), iv);
  sprintf(ciphertext, "%s", encrypted_bytes);
  return enc_length;
}
void setup() {
  Serial.begin(BAUD);
  Serial.setTimeout(60000);
  randomSeed(analogRead(5));
}

unsigned long loopcount = 0;
unsigned long loopEnd = 100;
unsigned long timeRecord = 0;
byte enc_iv_to[N_BLOCK] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

void loop() { //Serial.print("readBuffer length: "); Serial.println(readBuffer.length());
  if(loopcount==0){
    timeRecord = millis();
  }
  aes_init();
  memcpy(enc_iv_to, aes_iv, sizeof(aes_iv));
  int randomlength = (random()%(INPUT_BUFFER_LIMIT-1))+1;
  randstr(plaintext, randomlength);
  uint16_t len = encrypt_to_ciphertext(plaintext, enc_iv_to);
  loopcount++;
  memset(plaintext, 0, sizeof(plaintext));
  if(loopcount == loopEnd){
    time_t finishtime = millis();
    Serial.println(finishtime-timeRecord);
  }
  //Serial.print("Encrypted length = "); Serial.println(len);
  //Serial.println(loopcount);
  //Serial.println("---");
  if(loopcount>=loopEnd){
    while(1);
  } 
}