def print_progress_bar(iteration, endpoint, prefix='', suffix='', decimals=1, length=100, fill='#', printEnd="\r",
                       reverse=False, startpoint=0):
    """
    Print iterations progress
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = (iteration - startpoint) / (endpoint - startpoint) if endpoint != startpoint else 1
    percent_str = ("{0:." + str(decimals) + "f}").format(percent * 100)
    filledLength = int(length * percent)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent_str}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == endpoint:
        print()
