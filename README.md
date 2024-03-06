# HyGenie: Making Hands Cleaner

## The problem
The spread of germs and pathogens happens very quickly and without our knowledge. The COVID-19 Pandemic especially emphasized the importance of keeping our hands clean, as the virus spread quickly through contact with contaminated surfaces. Maintaining proper hygiene habits is very important, especially in public areas like airports, stores, to ensure that you don't fall ill or spread germs to others who may come in contact with you. The best way to keep our hands clean is by washing them properly. For this purpose, the World Health Organization (WHO) specified step-by-step guidelines to ensure that your hands are washed thoroughly. Additionally, we noticed that in public restrooms, notices are put up reminding people to wash their hands for a specified amount of time. That brings us to the question: How many people actually know about these these guidelines, and how many people actually pay attention to and follow the recommendations on the notices put up on campus and other public restrooms?

Inspired by this problem statement, we aimed to create an application that could be used in public restrooms, that tracked the hand movements of people washing their hands and showed them through a display, how long they have to perform the current washing action, which action they have to perform next, and if the user stops performing one of the actions before reaching 100% of the action time, it will stop the countdown and ask them to finish performing the action for the specified time. This ensures that all the steps specified by WHO to wash your hands are followed.

## Challenges we ran into

- Data Collection and Feature Extraction Challenges: After a deep dive into hand gesture datasets and recognition models we found that there are no other datasets that can be used for our specific use case. Therefore, we had to collect our own data points, as well as identify which of the 426 features were important for our classification and detection tasks. Some features like finger positions, which we thought were important in the beginning, ended up causing our model to identify incorrect relationships. At the end of the day, we found that for the three gestures we were tracking, the palm coordinates and palm normals for each hand were the relevant features for gesture recognition.

- We also had to learn how to implement real-time tracking of hand movements and gestures, and additionally tune it to detect the specific gesture being performed in real time. This was a bit tricky to navigate as we had to sample timestamps and frames intermittently but had to find the sweet spot so that we had enough information to correctly identify changes in gestures, but also making sure that the gesture recognition was not delayed.

## Technologies we used

- TensorFlow
- Websocket
- LeapMotion

## Authors

- Joseph Ajayi
- Vishnu Krishnan
- Manognya Bhattaram
- Mahanthesh R