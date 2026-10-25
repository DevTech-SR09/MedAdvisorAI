from .recommendation_agents import (
    diet_agent,
    exercise_agent,
    medication_agent,
    lifestyle_agent,
    cardiovascular_agent,
)


# ============================================================
# Recommendation Coordinator
# ============================================================

def generate_recommendations(health, risk_level):
    """
    Runs all specialized recommendation agents and coordinates
    their outputs into a unified personalized recommendation plan.

    Parameters:
        health:
            HealthAssessment object containing the patient's
            health information.

        risk_level:
            ML-generated cardiovascular risk level:
            LOW RISK
            MODERATE RISK
            HIGH RISK

    Returns:
        Dictionary containing:
            - risk_level
            - recommendations
            - total_recommendations
            - priority_order
    """

    # ========================================================
    # STEP 1 — RUN SPECIALIZED AGENTS
    # ========================================================

    diet_result = diet_agent(
        health=health,
        risk_level=risk_level
    )

    exercise_result = exercise_agent(
        health=health,
        risk_level=risk_level
    )

    medication_result = medication_agent(
        health=health,
        risk_level=risk_level
    )

    lifestyle_result = lifestyle_agent(
        health=health,
        risk_level=risk_level
    )

    cardiovascular_result = cardiovascular_agent(
        health=health,
        risk_level=risk_level
    )

    # ========================================================
    # STEP 2 — COLLECT ALL AGENT OUTPUTS
    # ========================================================

    agent_results = [
        diet_result,
        exercise_result,
        medication_result,
        lifestyle_result,
        cardiovascular_result
    ]

    # ========================================================
    # STEP 3 — REMOVE DUPLICATE RECOMMENDATIONS
    # ========================================================

    seen = set()

    coordinated_recommendations = []

    for agent in agent_results:

        unique_recommendations = []

        for recommendation in agent.get(
            "recommendations",
            []
        ):

            # Convert recommendation to lowercase
            # for duplicate comparison

            normalized = recommendation.strip().lower()

            if normalized not in seen:

                seen.add(normalized)

                unique_recommendations.append(
                    recommendation
                )

        # Create a copy of the agent result
        # so the original agent output is not modified

        coordinated_agent = agent.copy()

        coordinated_agent[
            "recommendations"
        ] = unique_recommendations

        coordinated_recommendations.append(
            coordinated_agent
        )

    # ========================================================
    # STEP 4 — DEFINE AGENT PRIORITY
    # ========================================================

    # Higher number = higher priority

    priority_map = {

        "HIGH RISK": {
            "Cardiovascular": 5,
            "Medication": 4,
            "Lifestyle": 3,
            "Diet": 2,
            "Exercise": 1
        },

        "MODERATE RISK": {
            "Cardiovascular": 5,
            "Lifestyle": 4,
            "Diet": 3,
            "Exercise": 2,
            "Medication": 1
        },

        "LOW RISK": {
            "Lifestyle": 5,
            "Diet": 4,
            "Exercise": 3,
            "Cardiovascular": 2,
            "Medication": 1
        }
    }

    # Default priority if an unexpected risk level occurs

    default_priority = {
        "Cardiovascular": 5,
        "Lifestyle": 4,
        "Diet": 3,
        "Exercise": 2,
        "Medication": 1
    }

    current_priority = priority_map.get(
        risk_level.upper(),
        default_priority
    )

    # ========================================================
    # STEP 5 — ASSIGN PRIORITY TO EACH AGENT
    # ========================================================

    for agent in coordinated_recommendations:

        category = agent.get(
            "category",
            agent.get("agent", "")
        )

        # Normalize category name

        category_name = str(
            category
        ).strip()

        priority = current_priority.get(
            category_name,
            1
        )

        agent["priority"] = priority

    # ========================================================
    # STEP 6 — SORT AGENTS BY PRIORITY
    # ========================================================

    coordinated_recommendations.sort(
        key=lambda x: x.get(
            "priority",
            1
        ),
        reverse=True
    )

    # ========================================================
    # STEP 7 — COUNT FINAL RECOMMENDATIONS
    # ========================================================

    total_recommendations = sum(
        len(
            agent.get(
                "recommendations",
                []
            )
        )
        for agent in coordinated_recommendations
    )

    # ========================================================
    # STEP 8 — CREATE PRIORITY ORDER
    # ========================================================

    priority_order = []

    for agent in coordinated_recommendations:

        if agent.get("recommendations"):

            priority_order.append({
                "agent": agent.get(
                    "agent",
                    "Recommendation Agent"
                ),
                "category": agent.get(
                    "category",
                    ""
                ),
                "priority": agent.get(
                    "priority",
                    1
                )
            })

    # ========================================================
    # STEP 9 — RETURN FINAL COORDINATED RESULT
    # ========================================================

    return {

        "risk_level": risk_level,

        "recommendations":
            coordinated_recommendations,

        "total_recommendations":
            total_recommendations,

        "priority_order":
            priority_order
    }
    