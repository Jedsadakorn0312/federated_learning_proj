def calculator(a,b,c):
    if c == "+":
        return a + b
    elif c == "-":
        return a - b
    elif c == "*":
        return a * b
    else:
        return "Invalid operator"

first_number=  (input("Enter first number: "))
second_number =  (input("Enter second number: "))
operator = input("Enter operator (+, -, *): ")

result = calculator(first_number, second_number, operator)
print("Result:", result)