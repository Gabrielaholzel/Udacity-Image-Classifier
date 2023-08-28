# First Project 
## Project Description
In this project, I used a created image classifier to identify dog breeds. 

## Context
In this hypothetical context, I had to contribute to the organization of a citywide dog show. My role involved assisting the organizing committee with contestant registration. To ensure the integrity of the competition, I needed to verify that every participant registering for the dog show was, in fact, registering a real dog and not another type of pet.

To achieve this, an existing Python classifier was implemented. This classifier was designed to analyze images submitted during registration and determine whether the registered pet was indeed a dog. If the classifier confirmed that the pet was a dog, the registration process continued. However, if the classifier detected that the pet was not a dog, it prompted the organizers to review the registration.

This project demonstrated the practical application of machine learning and image classification in real-world scenarios, showcasing how technology can be used to enhance event organization and decision-making processes.

## Principal Objectives
1) Correctly identify which pet images are of dogs (even if the breed is misclassified) and which pet images aren't of dogs.
2) Correctly classify the breed of dog, for the images that are of dogs.
3) Determine which CNN model architecture (ResNet, AlexNet, or VGG), "best" achieve objectives 1 and 2.
4) Consider the time resources required to best achieve objectives 1 and 2, and determine if an alternative solution would have given a "good enough" result, given the amount of time each of the algorithms takes to run.


## Walkthrough
The `check_images.py` is the program file that achieves the four objectives above. This file contains a `main()` function that outlines each step of the program. This function consists of the following steps.

The first thing to do is measure total program runtime by collecting start time:
```python
start_time = time()
```

Then, we define `get_input_args` function within the file `get_input_args.py`(description of file [here][cla]). This function retrieves three Command Line Arugments from user as input from the user running the program from a terminal window. This function returns the collection of these command line arguments from the function call as the variable `in_arg`.
```python
in_arg = get_input_args()
```
We pass the variable `in_arg` as an argument for the function `check_command_line_arguments`:
```python
check_command_line_arguments(in_arg)
```

Now, we call the `classify_images` function within the file `classify_images.py`. This function creates classifier labels with classifier function, compares labels, and adds these results to the results dictionary - `results`. You can find a more throughout description of this function inside the `classify_images.py` file. 
```python
classify_images(in_arg.dir, results, in_arg.arch)
```

We continue calling the function `check_classifying_images` that checks results dictionary using `results`. You can find this function inside the `print_functions_for_lab_checks.py` file. 
```python
check_classifying_images(results)
```

Now we call `adjust_results4_isadog` function within the file `adjust_results4_isadog.py`. This function adjusts the results dictionary to determine if classifier correctly classified images as 'a dog' or 'not a dog'. This demonstrates if the model can correctly classify dog images as dogs (regardless of breed)
```python
print_results(results, results_stats, in_arg.arch, True, True)
```

Once this is over, we measure total program runtime by collecting end time.
```python
end_time = time()
```

Finally, we compute the overall runtime in seconds and print it:
```python
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) 
          )
```
[//]: ()
[cla]: <https://github.com/Gabrielaholzel/Udacity-Image-Classifier/blob/main/First_Project/command_line_arguments.md#command-line-arguments>



