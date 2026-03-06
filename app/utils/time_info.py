import datetime

def get_time_information()-> str:
    now = datetime.datetime.now()
    return (
        f"Current  real-time INformation: \n"
        f"day: {now.strftime('%A')}\n"
        f"Date: {now.strftime('%d')}\n"
        f"Month: {now.strftime('%B')}\n"
        f"Year: {now.strftime('%Y')}\n"
        f"time: {now.strftime('%H')}:{now.strftime('%M')}:{now.strftime('%S')}"
    )
    