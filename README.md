# Credit Card Fraud Detection Through Isolation-Forest
Master Degree Thesis Data Analytics for Business



https://user-images.githubusercontent.com/48164716/230611789-5f5f55bb-07ab-48df-b50a-5fd0fba5b160.mp4



### Summary
> Developing an effective fraud detection system is a complex process that
> typically requires more than one technique. The majority of tools fully approved
> by regulators are explicit rules, often inconsistent with the fast pace of time and
> technology. Although simulated transactions are purely demonstrative, they are
> a valuable tool for experimenting new techniques and support the development
> of fraud detection models.
> Investigators should supervise the development
> of new rules and features to spot anomalies, and seek continuous feedback
> from users and ex former fraudsters. As part of a Data Driven Layer, Isolation
> Forest can help existing methods to perform better and avoid explicit rules that
> might be fooled by more experienced fraudsters. Creating Features related to
> customer spending and information about previous risks over time seems a key
> element also in difficult fraud scenarios and prevent a drift in performance. Class
> Imbalance seems also a problem for many supervised models, and isolation
> forest performed well and stayed consistent. Furthermore, the use of a flexible
> alert can be of a good use in moments when policy makers or management
> requires more attention or less friction in the payment system. Investing in data
> collection and cloud computing solutions can help minimize the latency between
> registration and validity of transactions. However, it is essential to ensure data
> protection and address the trade-off between compliance and efficiency.

### Cross validation Results
The model has been trained using different features and tested using a sliding window cross validation.
Each set of feature represent an area of investigation and the final metric of evaluation was the AUC.

<p align="center">
  <img  width="600" src="https://user-images.githubusercontent.com/48164716/230608542-a0d61bcc-d3d5-4c35-82c4-4a30bbadfe60.png" />
</p>

### Scenario validation 
The model has tested using different fraud scenarios.


   * Scenario 1: Any transaction whose amount is more than 220 is a fraud. This scenario is not inspired by a real-world scenario. Rather, it will provide an obvious fraud pattern that should be detected by any baseline fraud detector. This will be useful to validate the implementation of a fraud detection technique.

  * Scenario 2: Every day, a list of two terminals is drawn at random. All transactions on these terminals in the next 28 days will be marked as fraudulent. This scenario simulates a criminal use of a terminal, through phishing for example. Detecting this scenario will be possible by adding features that keep track of the number of fraudulent transactions on the terminal. Since the terminal is only compromised for 28 days, additional strategies that involve concept drift will need to be designed to efficiently deal with this scenario.

  * Scenario 3: Every day, a list of 3 customers is drawn at random. In the next 14 days, 1/3 of their transactions have their amounts multiplied by 5 and marked as fraudulent. This scenario simulates a card-not-present fraud where the credentials of a customer have been leaked. The customer continues to make transactions, and transactions of higher values are made by the fraudster who tries to maximize their gains. Detecting this scenario will require adding features that keep track of the spending habits of the customer. As for scenario 2, since the card is only temporarily compromised, additional strategies that involve concept drift should also be designed.

<p align="center">
  <img  width="600" src="https://user-images.githubusercontent.com/48164716/230724238-80985ba9-38af-4d0a-9012-bbe409570379.png" />
</p>




