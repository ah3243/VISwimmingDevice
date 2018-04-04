"""
    Module calculates the slope and intercept of any input lines
"""

def calcSlope(P1, P2):
    """Calculates the slope of a lines based on two points
    
    Arguments:
        P1 {tuple} -- first points x,y coordinates
        P2 {tuple} -- second points x,y coordinates
    
    Returns:
        float -- slope of the line
    """

    # Slope is y/x
    try:
        return (float(P2[1])-P1[1])/(float(P2[0])-P1[0])
    except ZeroDivisionError:
        print("Line is vertical")
        return None

def calcIntercept(slope, P1):
    """" Calculates the y intercept point based on slope and second point   
    Arguments:
        P1 {tuple} -- second points x,y coordinates
        slope {float} -- pre calculated slope
    
    Returns:
        int -- y intercept point
    """

    # Intercept is (b = y - mx)
    if slope != None:
        return P1[1] - slope*P1[0]
    else:
        print("Slope is None")
        return None
