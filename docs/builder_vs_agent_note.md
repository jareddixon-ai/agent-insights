# Outside builder or agent at one brokerage?

A decision note, not legal advice. Take the open questions to a lawyer.

## The choice
- **Outside builder:** you pitch as a vendor. The brokerage runs the tool on its own machine, under an NDA and a data agreement. You never hold their data.
- **Agent at one brokerage:** you work there under a broker (you are working toward a California salesperson license) and build the tool for them from inside.

## Conflict of interest
- **Scoring colleagues.** As an agent you would score people you compete with for listings and referrals, and you would see their results. Even with random agent codes, the broker knows who is who. An outside builder holds no key and no stake in any agent's result.
- **Competing for the same clients.** Seeing which agents win offers, and how, is an edge you would hold over your own colleagues.
- **Who you work for.** An agent's duties run to clients and to the broker. A vendor's duty runs to the brokerage under the contract.
- **Trust.** Agents tend to accept a tool from a neutral outsider more readily than from a peer who benefits from the scores.

## MLS rules
- MLS data is licensed to participants under the MLS's own terms. Many MLSs restrict who may receive closed-sale data and for what. A brokerage's own transaction files are a different thing, but they may carry MLS-sourced fields.
- The brokerage, not you, decides what data it may hand over. The NDA and data agreement should say the brokerage confirmed its MLS rules allow it.
- This tool never scrapes MLS or listing sites. It reads only what the brokerage maps from its own files.
- Being a licensee does not change these limits. It adds the broker's supervision over everything real estate related you do.

## Vendor or staff
- A vendor works under a contract with a scope, a data agreement, a deletion clause and an end date.
- An agent building it as part of the job is closer to staff: the brokerage may own the work, and the line between your job and your project blurs. Settle ownership of the code, the right to reuse the open-source engine, and who pays for what, in writing, before you start.
- Brokerages usually have their lawyer read a vendor agreement. They are less used to reading an agent's side project.

## Recommendation
Start as an outside builder with the pilot terms already in the kit (their machine, mapped columns only, random codes with the key kept by them, NDA first, deletion with a signed letter, never used for pay, commission splits, hiring or firing). Revisit the agent route after your license, and only with the broker's written terms on ownership and on what you may see.

## What a brokerage should check with its own lawyer
**Scoring independent-contractor agents.**
- Does measuring and coaching agents on performance change how their relationship is treated (contractor or employee)? In California, how much control a brokerage exercises can matter. Ask whether using a tool like this to direct agents affects that.
- Does your agent agreement allow the brokerage to measure results this way, and to use the data it holds?
- If the tool is ever used for pay, commission splits, hiring or firing, the risk changes. Our promise is that it is not used that way. The brokerage should decide its own policy and write it down.
- Could a score act as a stand-in for a protected characteristic (for example, through who gets which listings)? Ask about fair housing and employment rules.

**Telling or not telling agents.**
- What must the brokerage tell its agents about measuring them, and what may it leave out? The answer depends on the agent agreement, state law and the brokerage's own policies.
- If the brokerage tells agents, what exactly should it say, and when?
- If it does not tell them, what is the risk if agents learn about it later?
- The tool shows nothing to agents in either case. The Disclosure setting in `config/disclosure.json` only changes one sentence in the readout. What agents are told is the brokerage's decision.
