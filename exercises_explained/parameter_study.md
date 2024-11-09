## Parameter Study
Goal's to generate a parameter study for the nonstationary 10-armed testbed from exercise 2.5 (10_armed.py), where the performance of algorithms is measured as a function of their parameters.

#### Highlights

`constant step-size Epsilon-Greedy`

    q(a) <- q*(a) + N(0, sigma^2)

`update rule`

    Q(a) <- Q(a) + alpha * (R - Q(a))

---