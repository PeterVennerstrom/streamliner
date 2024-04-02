from statistics import mean


def due_diligence(fleet, financial_statement, extra_due_diligence=True):
    def _due_diligence(model_name):
        counterparty_risk, controls_over_financial_reporting = fleet[model_name](
            financial_statement
        )
        counterparty_score = counterparty_risk["scores"][0]
        controls_score = controls_over_financial_reporting["scores"][0]
        due_diligence_rating = mean([counterparty_score, controls_score])

        return due_diligence_rating

    due_diligence_rating = _due_diligence("due_diligence")

    if extra_due_diligence:
        due_diligence_rating = mean(
            [due_diligence_rating, _due_diligence("due_diligence_extra")]
        )

    return due_diligence_rating
