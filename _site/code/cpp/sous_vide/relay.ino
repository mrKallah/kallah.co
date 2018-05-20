int min_temp = 53;
int max_temp = 54;
int RELAY_PIN = 4;
int TEMP_PIN = 2;

#include <OneWire.h>
#include <DallasTemperature.h>

//One_Wire and DallasTemprature settings
#define ONE_WIRE_BUS_1 2
OneWire oneWire_in(ONE_WIRE_BUS_1);
DallasTemperature sensor_inhouse(&oneWire_in);

void setup(void)
{
    //setup for printing to terminal
    Serial.begin(9600);
    //set the output and input pins
    pinMode(RELAY_PIN, OUTPUT);
    pinMode(TEMP_PIN, INPUT);
    //starts the sensor reading
    sensor_inhouse.begin();
}

void loop(void)
{
    //reads teh tempratures
    sensor_inhouse.requestTemperatures();
    float temp = sensor_inhouse.getTempCByIndex(0);
    //prints the tempratures
    Serial.println(temp);
    //Turns on heating if temprature is too low
    if(temp < min_temp) {
      digitalWrite(RELAY_PIN, HIGH);
    }
    //turns off heating if temprature is too high
    if (temp > max_temp ){
      digitalWrite(RELAY_PIN, LOW);
    }
    //does nothing if temprature is right
}




