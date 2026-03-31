from pydantic import BaseModel

class LoanRequest(BaseModel):
    income: float
    loan_amount: float
    credit_score: float
