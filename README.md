# Predicting number of air passengers on a given US domestic flight

_Personal submission for academic machine learning competition originally posted on [RAMP](https://ramp.studio).\
Set problem can be found in the following repository : [ramp-kits/air_passengers](https://github.com/ramp-kits/air_passengers)_
\
\
<b>Authors of presented solution :</b> Jean-Julien Cordano & Benoît Grand ([github.com/grdbenoit](https://github.com/grdbenoit)).


## Introduction :
The data set was donated to us by an unnamed company handling flight ticket reservations. The data is thin, it contains
<ul>
<li> the date of departure
<li> the departure airport
<li> the arrival airport
<li> the mean and standard deviation of the number of weeks of the reservations made before the departure date
<li> a field called <code>log_PAX</code> which is related to the number of passengers (the actual number were changed for privacy reasons)
</ul>

The goal is to predict the <code>log_PAX</code> column. The prediction quality is measured by (Root Mean Square Error) RMSE. 

The data is limited, and external data such as location information has been added by joining external data sets.

## Variable selection and transformation :
### a. External variables added to data set :
<ul>
<li> <i>CloudCover</i> & <i>CloudCover_d</i>: Relative number to reflect cloud cover in departure and arrival locations.
<li> <i>Mean Wind SpeedKm/h</i> & <i>Mean Wind SpeedKm/h_d</i>: Mean Wind SpeedKm/h for departure and arrival locations.
<li> <i>Events</i> & <i>Events_d</i>: Notable weather events on day of travel in departure and arrival locations.
</ul>

### b. Data preprocessing :
One-Hot Encoding was used to discretize categorical columns and standard scaling was applied on all numerical columns.

## Prediction results :

| Model                          | RMSE  |
|--------------------------------|-------|
| Random Forest                  | 0.632 |
| Linear Regression              | 0.612 |
| Linear Reg. (w/ external data) | 0.624 |
| ANN (w/ external data)         | 0.280 |

The best model that was tested was a 6 layer artificial neural network (ANN), yielding an <b>RMSE of 0.280</b>. With this prediction model, we achieved the 2nd highest score in a field of 10 teams.

### Testing the model :

#### Install :

To run a submission and the notebook you will need the dependencies listed
in `requirements.txt`. You can install the dependencies with the
following command-line:

```bash
pip install -U -r requirements.txt
```

If you are using `conda`, an `environment.yml` file is provided for similar
usage.

#### Test our submission

To run our submission, use the `ramp-test` command line as follows:

```bash
ramp-test --submission neural_network
```

You can get more information regarding this command line:

```bash
ramp-test --help
```

#### To go further

You can find more information regarding `ramp-workflow` in the
[dedicated documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html)
