def unsafe_function():
    # Hardcoded password - should be flagged
    password = "admin123"
    
    # SQL injection vulnerability - should be flagged
    user_id = request.args.get('id')
    query = "SELECT * FROM users WHERE id = '" + user_id + "'"
    
    # Using eval - should be flagged
    eval(user_input)
    
    return password
