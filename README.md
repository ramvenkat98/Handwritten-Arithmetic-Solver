Usage:

To use the solver, first create a folder with the images of equations you want solved. 

Then on the command line, from the directory 'Math Expression Solver', you can run the command: 
"python predictionUI.py pathToFolder", where pathToFolder is the path to the folder where you have saved the images

E.g. if you run "python predictionUI.py sampleTests", the application will solve the expressions in the sample images provided

Details:

This program solves hand-written math expressions involving the symbols +, - x and fractions (including nested fractions). It recognizes the expression in the image and then gives a step-by-step solution. This was inspired by a previous term project video for CMU's 15-112 course that I found online (listed under References).

Future Plans:

- Add support for more operations, including brackets and exponentials
- Implement variable-based symbolic operations such as algebraic simplification, differentiation and integration

References:
1. Inspired by this video: https://www.youtube.com/watch?v=iX6qbuyKka4&t=13s
2. Uses the animation framework found in CMU's 15-112 course notes (cited in code)
3. Uses a framework for building a neural network in scikit-learn found online (cited in code)
