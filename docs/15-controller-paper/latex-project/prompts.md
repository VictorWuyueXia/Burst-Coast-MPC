# For this conversation
you will help me polish my paper draft, which is targeting IEEE LCSS format and style and control venues. You will use nature writing->control writing->nature polish->academic humanizer skills through out this project writing tasks as standard writing pipeline.
# The draft
The previous writing is in docs\15-controller-paper\latex-project\baddraft.tex, which contains a bad draft. My polished version is at docs\15-controller-paper\latex-project\main.tex, which I only polished from beginning to III.A, III.B and after is still the baddraft. 
# Our goal of rewrite
We are to improve the writing section by section to bring them up to publication-like quality. Do not worry to much about citation or previous works, for now we focus on motivation, defining the problem, and explain our method, and showing the preliminary results. Focus on better define the problem clearly and professionally, concisely, and mathematically accurate and fully defined. We want to use simple grammars and one streamlined concept motivation and introduction, avoid jumping between different concepts. 
# Rewrite range
We now focus on the intro section of the draft. At this stage it does not need to be a completed intro section, it should be an adviser facing writing introducing the project. Two paragraphs would suffice: 
1. overall research goal is a short-action and long-coasting controller with dynamic horizon, emphasizing on completing the task in limited number of control interventions, inspired by astronautics and marine path planning.
2. As a benchmark we built a rotary pendulum system, around which we plan to build such controller. As a prior, we analyzed its dynamics and built a closed-form heuristic controller to propose nominal actions. which we use reduced order representation of the physical state in energy terms, find the goal energy to inject, then find the torque that would inject such energy by both energy and phase.
We expect the new section to be same or less the current length.
# My notes:
- remember our reader only barely knows about what is the task and what we are up to, and we are introducing them to know the problems we are solving, not merely presenting a formulation.
- focus on explaining why some thing is the way it is, not just stating the facts.
- Avoid stating actual values or units as we want to keep the paper mathematical and transferrable to other tasks. Only include the actual values if they are a design choice which would stay true across tasks.
- Fully define the equations with intuitions following them immediately. Explain the intuition of the intermediate formulas. The current dynamics is already too complicated for most people to understand, but we still have to present with details and absolute rigor. So by thoughtful and walk the readers through the derivations.
- Use math wisely. Show all necessary accurate concepts that they need to know, but avoid swarming the readers with dense math, and only include in a simpler clearer easy to understand form. (hint: omit parts like 100 or normalization by only "normalize()" would help, other similar applies, also avoid too many similar variables with different subscribs (O_tau O_H,tau O_C,tau, etc.)
- Avoid "somebody et al" style reference throughout the project.





# This step:
No hastily committing to writings yet, use the skills to write down bullet points structures, including what to talk about and what/where to present math equations\








Do not use words like "capture" "window". Keep the writing in universal control theory terminologies -- if some terms are named specifically for this task and cannot be transferred to other controllers, it is very likely badly named.

# Now
updated main.tex now contains my polished version of section I and II. Read them as well as section III.A (including the independent section III paragraph), we will now rewrite the section III.A to bring it up to a good academic writing quality and standards 
# Our goal of rewrite
We are to improve the writing section by section to bring them up to publication-like quality. Focus on motivation, defining the problem, and explain our method, and showing the preliminary results. Focus on better define the problem clearly and professionally, concisely, and mathematically accurate and fully defined. We want to use simple grammars and one streamlined concept motivation and introduction, avoid jumping between different concepts. 
# Rewrite range
We now focus on the Method section III and III.A of the draft. It should become a publishable quality that does not need us to change further when later we turn this draft into a publication. Four paragraphs would suffice: 
1. The dynamics and governing equation
2. The lagrangian and euler-lagrangian
3. The energy state and the energy governing equation
4. The jacobian of energy governing equation
5. The feasibility of actuations and states
We expect the new section to be roughly the same of the current length or slightly longer only if necessary.
# My notes:
- remember our reader only barely knows about what is the task and what we are up to, and we are introducing them to know the problems we are solving and our novel methods, not merely presenting what we did.
- focus on explaining why some thing is the way it is, not just stating the facts.
- Avoid stating actual values or units as we want to keep the paper mathematical and transferrable to other tasks or configurations. Only include the actual values if they would stay true across tasks.
- Fully define the equations with intuitions following them immediately. Explain the intuition of the intermediate formulas. The current dynamics is already too complicated for most people to understand, but we still have to present with details and absolute rigor. So by thoughtful and walk the readers through the derivations.
- Use math wisely. Show all necessary accurate concepts that they need to know, but avoid swarming the readers with dense math, and only include in a simpler clearer easy to understand form. (hint: omit terms like 100 for computing percentage, express normalization by only "normalize()" would help, other similar applies, also avoid too many similar variables with different subscribs (O_tau O_H,tau O_C,tau, etc.)
- Avoid "somebody et al" style reference throughout the project.
# This step:
No hastily committing to writings yet, use the skills pipeline to write down bullet points structures, including what to talk about and what/where to present math equations.


Now commits the plan in writing and rigorously adhere to our requirements and standards and style, using the skills pipeline. 
remember the wording rule and the words to avoid using