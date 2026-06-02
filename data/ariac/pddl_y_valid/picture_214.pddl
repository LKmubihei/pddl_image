(define (problem picture_213)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_pump red_regulator - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear red_pump)
        (clear red_regulator)
        (part_at red_pump pump_placement)
        (part_at red_regulator table)
    )
    
    (:goal
        (and
            (part_at red_regulator regulator_placement)
        )
    )
)