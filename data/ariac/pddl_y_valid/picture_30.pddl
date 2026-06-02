(define (problem picture_30)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_regulator green_regulator blue_pump - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear green_regulator)
        (clear red_regulator)
        (part_at green_regulator table)
        (on red_regulator blue_pump)
        (part_at blue_pump table)
    )
    
    (:goal
        (and
            (part_at blue_pump pump_placement)
            (part_at red_regulator regulator_placement)
        )
    )
)
