import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config import SMTP_EMAIL, SMTP_PASSWORD

logger = logging.getLogger(__name__)


def send_onboarding_email(
    student_name: str,
    receiver_email: str,
) -> bool:
    try:
        msg = MIMEMultipart()

        msg["From"] = SMTP_EMAIL
        msg["To"] = receiver_email
        msg["Subject"] = "Gradious Onboarding Form"

        body = f"""
Hello {student_name},

Thank you for your interest in Gradious.

Please complete your onboarding form using the link below:

https://forms.gle/HKjhCtu1ChZDayJD7

Once submitted, our admissions team will contact you regarding the next steps.

Regards,
Gradious Admissions Team
"""

        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()

        server.login(
            SMTP_EMAIL,
            SMTP_PASSWORD
        )

        server.send_message(msg)
        server.quit()

        logger.info(
            f"Onboarding email sent to {receiver_email}"
        )

        return True

    except Exception as e:
        logger.error(
            f"Failed sending onboarding email: {e}",
            exc_info=True
        )
        return False
